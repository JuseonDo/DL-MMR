from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import gc
from tqdm import tqdm
from typing import Tuple, List

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def inference(
        model:AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        prompts:list[str],
        batch_size:int = 16,
        **kwargs,
) -> list[str]:
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]
            generated_texts += process_batch(model,tokenizer,batch,**kwargs)

    generated_texts = [gen_text.replace(prompt,"").replace("$}}%","") \
                       for gen_text,prompt in zip(generated_texts,prompts)]

    return generated_texts

def process_batch(
        model:AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        batch:list[str],
        **kwargs,
) -> list[str]:
    model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    try:
        model_outputs = model.generate(**model_inputs,**kwargs)
        generated_texts = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    except KeyboardInterrupt as ke:
        print(ke)
        exit()
    except RuntimeError as re:
        print(re)
        if "CUDA" in str(re):
            import gc
            del model_inputs
            clear_memory()

            temp_batch_size = len(batch)//2
            print("temp_batch_size:",temp_batch_size)

            temp_batch_1 = batch[:temp_batch_size]
            generated_text_1 = process_batch(model,tokenizer,temp_batch_1,**kwargs)

            temp_batch_2 = batch[temp_batch_size:]
            generated_text_2 = process_batch(model,tokenizer,temp_batch_2,**kwargs)

            generated_texts = generated_text_1 + generated_text_2

    return generated_texts