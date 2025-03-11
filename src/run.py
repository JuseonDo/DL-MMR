import fire
import torch
import json
from transformers import BitsAndBytesConfig
from transformers import(
    AutoTokenizer,
    LlamaForCausalLM,
)

from src.inference_utils import inference
from src.utils.template_utils import apply_template

def parse_numbers(sample):
    return list(map(int, sample.split(', ')))

def main(
        given_test_pool_path:str,
        exemplar_pool_path:str,
        retrieved_idxs_path:str,
        model_size:str = "13",
        batch_size:int = 10,
):  
    model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"

    print("Loading Model...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=nf4_config,
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=True,
                                              padding_side="left",
                                              )
    tokenizer.pad_token = tokenizer.eos_token

    template = "Sentence:\n{src}\nSummary of the sentence without the less important words would be:\n"

    with open(given_test_pool_path) as f:
        test_dataset = [json.loads(line.strip()) for line in f]
    with open(exemplar_pool_path) as f:
        exemplar_pool = [json.loads(line.strip()) for line in f]
    with open(retrieved_idxs_path) as f:
        retrieved_idxs = [parse_numbers(line.strip()) for line in f]

    prompts = apply_template(
        test_dataset=test_dataset,
        exemplar_pool=exemplar_pool,
        retrieved_idxs=retrieved_idxs,
        template=template
    )

    generated_text = inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=200,
        do_sample=False,
    )

if __name__ == '__main__':
    with torch.no_grad():
        fire.Fire(main)