CUDA_VISIBLE_DEVICES=1 python ./src/retriever.py \
    --save_path ./retrieved_idx/idx \
    --distance_path ./faiss_distances/distance.json \
    --length_path ./metadata/bnc.tgt \
    --lambda_ 0.9 \
    --outlier_path ./metadata/bnc.outlier \