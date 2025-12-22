import model_utils
import rotation_utils
import quant_utils
import hadamard_utils
import utils
from transformers import AutoTokenizer

def main():
    print("Hello from quarot-re-implementation-on-llama-3-2-1b!")

    model = model_utils.get_model('meta-llama/Llama-3.2-1B', hf_token='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')
    model.eval()

    text = 'how are you'
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    input = tokenizer(text, return_tensors='pt').to(device='cpu')

    output = model.generate(**input,
                   pad_token_id=tokenizer.eos_token_id,
                   max_new_tokens=4,
                   output_scores=True,
                   return_dict_in_generate=True)
    output_decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f'output is {output_decoded}')


    print('Rotating the output')
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model)
    utils.cleanup_memory(verbos=True)
        
    quant_utils.add_actquant(model) #Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = True
        if 'o_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = True

    
    



if __name__ == "__main__":
    main()
