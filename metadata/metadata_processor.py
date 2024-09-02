import json
import fire
from metadata.utils.functions import (
    calcuate_cr,
    calculate_length,
    get_outlier_idx
)

def store_metadata(
        exemplar_pool_path:str,
        save_path:str,
):
    """
    
    """
    with open(exemplar_pool_path) as f:
        exemplar_pool = [json.loads(line.strip()) for line in f]
    sentences = [line['text'] for line in exemplar_pool]
    summaries = [line['summaries'][0] for line in exemplar_pool]

    src_lenths = calculate_length(sentences)
    tgt_lenths = calculate_length(summaries)
    crs = calcuate_cr(sentences, summaries)
    outlier_idxs = get_outlier_idx(crs, threshold=0.1)
    
    with open(save_path + '.src', 'w') as f:
        f.write('\n'.join(map(str,src_lenths)))
    with open(save_path + '.tgt', 'w') as f:
        f.write('\n'.join(map(str,tgt_lenths)))
    with open(save_path + '.crs', 'w') as f:
        f.write('\n'.join(map(str,crs)))
    with open(save_path + '.outlier', 'w') as f:
        f.write('\n'.join(map(str,outlier_idxs)))

if __name__ == '__main__':
    fire.Fire(store_metadata)