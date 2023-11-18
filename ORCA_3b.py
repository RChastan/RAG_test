import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Hugging Face model_path
model_path = 'psmathur/orca_mini_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

#%%

#generate text function
def generate_text(system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'

#%%

# Sample Test Instruction Used by Youtuber Sam Witteveen https://www.youtube.com/@samwitteveenai
# system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'
# instruction = 'how to cook proper bolognese ?'
# print(generate_text(system, instruction))

system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'\
    'Answer based on the following context:'\
    "2.3 Max-pooling layer\n"\
    "The biggest architectural diference between our implementation and the CNN of LeCun et al. (1998) is the use of a max-pooling layer instead of a sub-sampling layer. No such layer is used by Simard et al. (2003) who simply skips nearby pixels prior to convolution, instead of pooling or averaging. Scherer et al. (2010) found that max-pooling can lead to faster convergence, select superior invariant features, and improve generalization. The output of the max-pooling layer is given by the maximum activation over non-overlapping rectangular regions of size ( Kx,Ky). Maxpooling enables position invariance over larger local regions and downsamples the input image by a factor of KxandKyalong each direction. Technical Report No. IDSIA-01-11 3\n"\
    "2.4 Classication layer\n"\
    "Kernel sizes of convolutional lters and max-pooling rectangles as well as skipping factors are chosen such that either the output maps of the last convolutional layer are downsampled to 1 pixel per map, or a fully connected layer combines the outputs of the topmost convolutional layer into a 1D feature vector. The top layer is always fully connected, with one output unit per class label.\n"\
        
instruction = 'Who proposed to use max-pooling to increase convergence rate ?'
print(generate_text(system, instruction))