import model_utils
import rotation_utils
import quant_utils
import hadamard_utils
import utils
import torch
from transformers import AutoTokenizer

def main():
    print("Hello from quarot-re-implementation-on-llama-3-2-1b!")
    model_name = 'meta-llama/Llama-2-7b-hf'
    # model_name = 'meta-llama/Llama-3.2-1B'

    model = model_utils.get_model(model_name, hf_token='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')
    model.eval()

    text = 'how are you'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = tokenizer(text, return_tensors='pt').to(device='cpu')

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # print(f"Module type: {type(model).__name__}")
            activation[name+'_input'] = input
            activation[name] = output.detach()
        return hook
    
    lm_head_hook = model.lm_head.register_forward_hook(get_activation('lm_head'))
    print(model.model.named_modules)

    # output = model.generate(**input,
    #                pad_token_id=tokenizer.eos_token_id,
    #                max_new_tokens=1,
    #                output_scores=True,
    #                return_dict_in_generate=True)
    # output_decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = model(**input, pad_token_id=tokenizer.eos_token_id)
    print(f'output is {output.logits} and shape is {output.logits.shape}')

    lm_head = activation['lm_head']
    target = lm_head


    print('Rotating the output')
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model)
    utils.cleanup_memory(verbos=True)
        
    # quant_utils.add_actquant(model) #Add Activation Wrapper to the model
    # qlayers = quant_utils.find_qlayers(model)
    # for name in qlayers:
    #     if 'down_proj' in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    #         qlayers[name].online_full_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].fp32_had = True
    #     if 'o_proj' in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
    #         qlayers[name].online_partial_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
    #         qlayers[name].fp32_had = True


    # output = model.generate(**input,
    #                pad_token_id=tokenizer.eos_token_id,
    #                max_new_tokens=1,
    #                output_scores=True,
    #                return_dict_in_generate=True)
    # output_decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = model(**input, pad_token_id=tokenizer.eos_token_id)
    print(f'rotated output is {output.logits} and shape is {output.logits.shape}')

    lm_head = activation['lm_head']
    predictions = lm_head

    mse = torch.nn.functional.mse_loss(predictions, target)
    print(f'mse is {mse}')

    
    



if __name__ == "__main__":
    main()
