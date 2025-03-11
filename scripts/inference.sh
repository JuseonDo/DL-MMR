CUDA_VISIBLE_DEVICES=1 python ./src/run.py \
    --exemplar_pool_path ./dataset/BNC/bnc_test.jsonl \
    --given_test_pool_path ./dataset/Broadcast/broadcast_test.jsonl \
    --retrieved_idxs_path ./retrieved_idx/idx \