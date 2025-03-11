from typing import List
from nltk import word_tokenize

def calculate_length(sentences:List[str]):
    return [len(sentence.split()) for sentence in sentences]

def calcuate_cr(sentences:List[str], summaries:List[str]):
    assert len(sentences) == len(summaries)
    return [
        len(word_tokenize(summary)) / len(word_tokenize(sentence)) for sentence, summary in zip(sentences, summaries)
    ]

def isOutlier(cr, threshold:float):
    return not (threshold < cr < 1 - threshold)

def get_outlier_idx(crs:List[float], threshold:float = 0.1):
    return [idx for idx, cr in enumerate(crs) if isOutlier(cr, threshold)]
