import os
import torch
import fire
from tqdm import tqdm
from src.rag_utils.retriever_utils import (
    get_distances,
    get_lengths,
    get_outliers,
    retrieving,
    dist_scaling,
    length_scaling,
)

def main(
        save_path:str,
        distance_path:str,
        length_path:str,
        lambda_:float,
        outlier_path = None,
):
    device = "cuda"
    distances = get_distances(distance_path).to(device)
    number_of_test = distances.size()[0]
    number_of_train = distances.size()[1]

    print("number_of_test:",number_of_test)
    print("number_of_train:",number_of_train)

    if outlier_path is not None:
        outliers,outliers_idxs = get_outliers(
            outlier_path,
            number_of_train
        )
        outliers = outliers.to(device)
    else:
        outliers = torch.tensor([0]*number_of_train, dtype=torch.float32).to(device)
        outliers_idxs = []
        
    distances = dist_scaling(distances=distances,outlier_idxs=outliers_idxs).to(device)
    
    lengths = get_lengths(length_path).to(device)
    lengths = length_scaling(lengths=lengths,outlier_idxs=outliers_idxs).to(device)

    for distance in tqdm(distances):
        idxs = retrieving(
            distance=distance,
            lengths=lengths,
            lambda_=lambda_,
            outlier=outliers
        )
        with open(save_path,'a') as f:
            f.write(', '.join(list(map(str,idxs)))+'\n')

if __name__ == '__main__':
    fire.Fire(main)