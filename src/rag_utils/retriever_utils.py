import fire
import torch
import json
import os
from tqdm import tqdm
from typing import List, Tuple

def get_distances(
        dist_path:str,
) -> torch.tensor:
    with open(dist_path) as f:
        distances = [
            torch.tensor(json.loads(line.strip())['scores'],dtype=torch.float16) for line in f
        ]
    return torch.stack(distances)

def dist_scaling(distances:torch.tensor,outlier_idxs:List[int]):
    remaining_columns = [idx for idx in range(distances.size(1)) if idx not in outlier_idxs]
    filtered_distances = distances[:, remaining_columns]
    MIN = torch.min(filtered_distances)
    MAX = torch.max(filtered_distances)
    RANGE = MAX - MIN
    return (distances - MIN)/RANGE

def length_scaling(lengths:torch.tensor,outlier_idxs:List[int]):
    remaining_columns = [idx for idx in range(lengths.size(0)) if idx not in outlier_idxs]
    filtered_lens = lengths[remaining_columns]
    MIN = torch.min(filtered_lens)
    MAX = torch.max(filtered_lens)
    RANGE = MAX - MIN
    return (lengths - MIN)/RANGE


def get_lengths(
    length_path:str,
) -> torch.tensor:
    with open(length_path) as f:
        lengths = torch.tensor([float(line.strip()) for line in f],dtype=torch.float16)
    return lengths


def get_outliers(
        outlier_path:str,
        number_of_train:int,
) -> Tuple[torch.tensor, torch.tensor]:
    with open(outlier_path) as f:
        outlier_idxs = [int(line.strip()) for line in f]
    outlier_tensor = [0]*number_of_train
    for out_idx in outlier_idxs:
        outlier_tensor[out_idx] = 1000
    return torch.tensor(outlier_tensor,dtype=torch.float32),outlier_idxs

def smallest_indices(arr:torch.tensor) -> int:
    return torch.sort(arr)[1][0].item()


def retrieving(
        distance:torch.tensor,
        lengths:torch.tensor,
        lambda_:float,
        outlier:torch.tensor,
        number_of_examplars:int = 8,
) -> List[int]:
    selected_examplars_idx = []
    selected_examplars_len = []
    
    for _ in range(number_of_examplars):
        if len(selected_examplars_len) > 0:
            min_diff = lambda_ * torch.min(torch.abs(lengths.unsqueeze(0) - torch.tensor(selected_examplars_len).unsqueeze(1).to('cuda')),dim=0)[0]
        else:
            min_diff = torch.zeros(len(lengths)).to('cuda')
        distance = (1 - lambda_) * distance.to('cuda') if len(selected_examplars_idx) > 0 else distance.to('cuda')

        em_score = distance.to('cuda') - min_diff.to('cuda')
        em_score = em_score + outlier
        for si in selected_examplars_idx:
            em_score[si] += 1000

        idx = smallest_indices(em_score)
        selected_examplars_idx.append(idx)
        selected_examplars_len.append(lengths[idx])
    return selected_examplars_idx