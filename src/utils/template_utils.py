from typing import List


def apply_template(
        test_dataset:List[dict],
        exemplar_pool:List[dict],
        retrieved_idxs:List[List[int]],
        template:str
):
    """
    test_dataset:
        [
            {
                'text': sentence,
            },
            {
                'text': sentence,
            },
        ]
    exemplar_pool:
        [
            {
                'text': sentence,
                'summaries': [summary, summary, ...]
            },
            {
                'text': sentence,
                'summaries': [summary, summary, ...]
            },
        ]
    retrieved_idxs:
        [
            [
                571, 786, 220, 432, 389, 74, 19, 49
            ],
            [
                577, 786, 590, 432, 387, 68, 19, 394
            ],
        ]
    """

    prompts = []
    for test_item, idxs in zip(test_dataset, retrieved_idxs):
        exemplar_items = [exemplar_pool[idx] for idx in idxs]
        
        shots = [
            template.format(src=exemplar_item['text']) + exemplar_item['summaries'][0]  for exemplar_item in exemplar_items
        ]

        prompt = '\n'.join(shots) + '\n' + template.format(src=test_item['text'])
        prompts.append(prompt)
        
    return prompts