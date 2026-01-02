# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoConfig
import torch
from torch.export import export
import tqdm
import rotation_utils
import typing

def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

def fuse_layernorm():
    for layer in model.model.layers:
            
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        
        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)
                        
        
    fuse_ln_linear(model.model.norm, [model.lm_head])
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)

def rotate_embeddings():
    W = model.model.embed_tokens.weight.data
    W_ = W.to(device='cpu', dtype=torch.float64) @ Q.to(device='cpu', dtype=torch.float64)
    model.model.embed_tokens.weight.data = W_

def rotate_head():
    W = model.lm_head.weight.data
    W_ = W.to(device='cpu', dtype=torch.float64) @ Q.to(device='cpu', dtype=torch.float64)
    model.lm_head.weight.data = W_

def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device='cpu', dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q.to(dtype=torch.float64)).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device='cpu', dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T.to(dtype=torch.float64), W_).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device='cpu', dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q.to(dtype=torch.float64)).to(device="cpu", dtype=dtype)
    
def rotate_mlp_output(layer, Q):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device='cpu', dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T.to(dtype=torch.float64), W_).to(device="cpu", dtype=dtype)

def rotate_offline_weights():
    layers = model.model.layers
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q)
        rotate_attention_output(layers[idx], Q)
        rotate_mlp_input(layers[idx], Q)
        rotate_mlp_output(layers[idx], Q)



model_name = 'meta-llama/Llama-3.2-1B'

config = AutoConfig.from_pretrained(model_name)
if config.tie_word_embeddings:
    config.tie_word_embeddings = False


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    torch_dtype='auto',
    token='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',
)
model.eval()

Q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size)
model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

prompt = "what is my name?"
inputs = tokenizer(prompt, return_tensors='pt')

exported_program = export(
    model,
    args=(inputs.input_ids, inputs.attention_mask),
    strict=True
)


for node in exported_program.graph.nodes:
    # if 'input_layernorm_weight' in node.name:
    #     # Replace the original aten.embedding.default layer with a new one
    #     # new_node = exported_program.graph.call_function(
    #     #     torch.sum, node.args, node.kwargs
    #     # )
    #     # node.replace_all_uses_with(new_node)
    #     # exported_program.graph.erase_node(node)

    #     print(f'name : {node.name} and operation : {node.op} and target : {node.target}')
    #     print(f'args : {node.args}')
    #     name = exported_program.state_dict['model.norm.weight']
    #     print(f'{name} and type {type(name)}\n\n\n\n\n')
    #     # print(f'{exported_program.graph_signature}')

    #     break

    # elif 'post_attention_layernorm_weight' in node.name:
    #     # print(f'name of the node is {node.name}')
    #     pass
    if node.op == 'call_function':
        exported_program.graph.inserting_before(node) # insert the node after all of the placeholders in the graph. Thus this is the first call_function in the graph
        node  = exported_program.graph.create_node('call_function', fuse_layernorm, name='fuse_layernorm')
        print(f'fuse_layernorm node has been inserted')

        exported_program.graph.inserting_after(node)
        node  = exported_program.graph.create_node('call_function', rotate_embeddings, name='rotate_embeddings')
        print(f'rotate_embeddings node has been inserted')

        exported_program.graph.inserting_after(node)
        node  = exported_program.graph.create_node('call_function', rotate_head, name='rotate_head')
        print(f'rotate_head node has been inserted')

        exported_program.graph.inserting_after(node)
        node  = exported_program.graph.create_node('call_function', rotate_offline_weights, name='rotate_offline_weights')
        print(f'rotate_offline_weights node has been inserted')
        break

exported_program.graph.lint() # Validate the graph
exported_program.graph_module.recompile() # Recompile the graph
print('Linting and recompiling successful')
print(exported_program.graph.print_tabular())

output = model(**inputs, pad_token_id=tokenizer.eos_token_id)
print(f'\n\n\n pre rotated graph output is : {output.logits}')

converted_model_response = exported_program.module()(inputs.input_ids, inputs.attention_mask)
print(f'\n\n\n converted graph output is {converted_model_response}')

target = output.logits
predictions = converted_model_response.logits
mse = torch.nn.functional.mse_loss(predictions, target)
print(f'mse is {mse}')



# # print(exported_program.graph_module.print_readable()) # prints code along with arguments and their values
# torch.export.save(exported_program, "llama3_2_1B.pt2", pickle_protocol=4)