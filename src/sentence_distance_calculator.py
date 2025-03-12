import fire
import json
from src.faiss_utils import FaissSearch
import torch
import gc
from tqdm import tqdm


def write_dicts_to_jsonl(dictionary, file_path):
    with open(file_path, 'a') as file:
        json.dump(dictionary, file)
        file.write('\n')

def main(
        exemplar_pool_path:str,
        given_test_pool_path:str,
        save_path:str,
        batch_size:int = 16,
        faiss_batch_size:int = 200,
        model_name:str = "facebook/bart-large",
):
    """
        exemplar pool json file
            {
                'text': sentence,
                'summaries': [summary, ...],
            },
            {
                'text': sentence,
                'summaries': [summary, ...],
            },
    """
    with open(exemplar_pool_path) as f:
        train_dataset = [json.loads(line.strip()) for line in f]
    train_src = [td['text'] for td in train_dataset]

    with open(given_test_pool_path) as f:
        test_dataset = [json.loads(line.strip()) for line in f]
    test_src = [td['text'] for td in test_dataset]

    NUMBER_OF_TRAIN = len(train_src)
    NUMBER_OF_TEST = len(test_src)

    print("NUMBER_OF_TRAIN:",NUMBER_OF_TRAIN)
    print("NUMBER_OF_TEST:",NUMBER_OF_TEST)


    faiss_search = FaissSearch(
        model_name_or_path = model_name,
        tokenizer_name_or_path = model_name,
        train_data=train_src,
        test_data=test_src,
        device="cuda",
        batch_size=batch_size,
        similarity_type = "distance",
    )


    gc.collect()
    torch.cuda.empty_cache()

    for i in tqdm(range(0, len(faiss_search.test_embedd), faiss_batch_size)):
        if i+faiss_batch_size < len(faiss_search.test_embedd):
            batch = faiss_search.test_embedd[i:i+faiss_batch_size]
        else:
            batch = faiss_search.test_embedd[i:]
        Distances, Indexs = faiss_search.faiss_index.search(batch, NUMBER_OF_TRAIN)

        for dists, idxs in zip(Distances, Indexs):
            scores = [0]*NUMBER_OF_TRAIN
            for d, i in zip(dists, idxs):
                scores[i] = float(d)
            write_dicts_to_jsonl({"scores":scores},save_path)

if __name__ == "__main__":
    fire.Fire(main)