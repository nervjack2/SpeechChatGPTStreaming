import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser

PROMPT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a chatbot, only response precisely. Modality: {{User: {}, Machine: {}}}.{}<|eot_id|>\n"
HEADER_TEMPLATE = "<|start_header_id|>{}<|end_header_id|>\n\n"

def process_input(args, tokenizer):
    speech_style = "" if args.user_modal == args.machine_modal == "text" else " Speech Style: Audio Book."
    input_text = PROMPT_TEMPLATE.format(args.user_modal, args.machine_modal, speech_style)
    with open(args.input_txt) as f:
        input_text += HEADER_TEMPLATE.format('User') + f.readline().strip() + '<|eot_id|>'
    input_text += HEADER_TEMPLATE.format('Machine')
    inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
    return inputs

def generate(args, LM, tokenizer, past_key_values=None):
    inputs = process_input(args, tokenizer)
    generation_output = LM.generate(**inputs, return_dict_in_generate=True, past_key_values=past_key_values, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    ### The following information might be useful for multi-turn generation ###
    # past_key_values = generation_output['past_key_values'] (kv cache)
    # inputs['input_ids'] = generation_output['sequences']
    # inputs['attention_mask'] = inputs['input_ids'].new_ones(inputs['input_ids'].size())
    result_ids = generation_output['sequences'][0][inputs['input_ids'].size(1):]

    return result_ids, generation_output

def post_process(result_ids, tokenizer):
    results = tokenizer.decode(result_ids).replace('|>', '|> ').replace('<|', ' <|').replace('  ', ' ').split(' ')
    kms = []
    words = []
    for u in results:
        if u == "<|eot_id|>":
            break
        if u[:2] == '<|' and u[-2:] == '|>':
            kms.append(u[2:-2])
        else:
            words.append(u)
    return kms, words
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_txt", type=str, default="tmp.txt", help="Input unit file")
    parser.add_argument("--user_modal", type=str, default="speech", choices=["speech", "text"], help="Modal of user input")
    parser.add_argument("--machine_modal", type=str, default="speech", choices=["speech", "text"], help="Modal of machine output")
    parser.add_argument("--model_ckpt", type=str, default="./llama_ckpt", help="HF model checkpoint")
    parser.add_argument("--output_path", type=str, default="tmp.jsonl", help="Output file (in jsonl format)")
    parser.add_argument("--device", type=str, default="cuda", help="Acceleration device")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max number of generated tokens")
    args = parser.parse_args()
    
    # initialize the model and the tokenizer
    LM = AutoModelForCausalLM.from_pretrained(args.model_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    
    # generate and post process
    result_ids, _ = generate(args, LM, tokenizer)
    kms, words = post_process(result_ids, tokenizer)

    # write output
    with open(args.output_path, 'w') as f:
        f.write(json.dumps({'code': ' '.join(kms), 'text': ' '.join(words).strip()}, ensure_ascii=False) + '\n')