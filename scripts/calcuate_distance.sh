CUDA_VISIBLE_DEVICES=1 python ./src/sentence_distance_calculator.py \
    --given_test_pool_path ./dataset/Broadcast/broadcast_test.jsonl \
    --exemplar_pool_path ./dataset/BNC/bnc_test.jsonl \
    --save_path ./faiss_distances/distance.json \
