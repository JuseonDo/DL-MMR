# Considering Length Diversity in Retrieval-Augmented Summarization

* Authors: Juseon-Do, Jaesung Hwang, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura
* Paper Link: [DL-MMR](https://arxiv.org/abs/2406.11097)


## Structure
```
DL-MMR
|
├── dataset
│   ├── Google
│   │   ├──google_test.jsonl
│   │   ├──google_valid.jsonl
│   │   └──google_train.jsonl
│   ├── Broadcast
│   │   └──broadcast_test.jsonl
│   └── BNC
│       └──bnc_test.jsonl
|   
├── scripts
|
└── src


```

## Run
```
$ cd DL-MMR
$ bash scripts/save_metadata.sh
$ bash scripts/calcuate_distance.sh
$ bash retrieve.sh
$ inference.sh
```


# Evaluation
The metrics used in this work are listed in [evaluation_metrics](https://github.com/JuseonDo/InstructCMP/blob/main/src/evaluate_utils/evaluate_functions.py). For each metric, we have steps.txt which presents the steps to setup and run the metric.
# Contact
If you have any questions about this work, please contact **Juseon-Do** using the following email addresses: **dojuseon@gmail.com** or **doju00@naver.com**. 

