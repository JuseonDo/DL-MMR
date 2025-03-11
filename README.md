# Considering Length Diversity in Retrieval-Augmented Summarization

* Authors: Juseon-Do, Jaesung Hwang, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura
* Paper Link: [DL-MMR]("")


## Structure
```
DL-MMR
|
├── dataset
│   ├── Google
│   ├── Broadcast
│   └── BNC
|   
├── scripts
|
└── src
    └── faiss_utils
    └── inference_utils
    └── rag_utils


```

## Run
```
$ cd DL-MMR
$ bash scripts/save_metadata.sh
$ bash scripts/calcuate_distance.sh
$ bash scripts/retrieve.sh
$ bash scripts/inference.sh
```


# Evaluation
The metrics used in this work are in [evaluation_metrics](https://github.com/JuseonDo/InstructCMP/evaluation).

```python
post_processed_outputs = generated_output_post_processing(generated_text)
result = evaluate(targets, sources, post_processed_outputs)
```

# Contact
If you have any questions about this work, please contact **Juseon-Do** using the following email addresses: **dojuseon@gmail.com** or **doju00@naver.com**. 

