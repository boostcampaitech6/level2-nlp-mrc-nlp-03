#!/usr/bin/sh
# Model Arguments 설명
# --model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models

# Data Training Arguments 설명
# --dataset_name: The name of the dataset to use.
# --data_path: The path of the 'data' directory.
# --context_path: The name of the documents to retrieve.
# --overwrite_cache: Overwrite the cached training and evaluation sets
# --max_seq_length: The maximum total input sequence length after tokenization.
# --pad_to_max_length: Whether to pad all samples to `max_seq_length`.
# --doc_stride: When splitting up a long document into chunks, how much stride to take between chunks.
# --max_answer_length: The maximum length of an answer that can be generated.
# --eval_retrieval: Whether to run passage retrieval using sparse embedding.
# --num_clusters: Define how many clusters to use for faiss.
# --top_k_retrieval: Define how many top-k passages to retrieve based on similarity.
# --use_faiss: Whether to build with faiss
# --bm25: Whether to use BM25

# Training Arguments (HuggingFace: https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/trainer#transformers.TrainingArguments)
# --output_dir: The output directory where the model predictions and checkpoints will be written.
# --overwrite_output_dir: If True, overwrite the content of the output directory.
# --do_eval: Whether to run evaluation on the validation set or not.
# --do_predict: Whether to run predictions on the test set or not.
# --seed: Random seed that will be set at the beginning of training.
# 추가적인 설정은 HuggingFace 참고 (HuggingFace: https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/trainer#transformers.TrainingArguments)

python3 code/inference.py \
    --model_name_or_path code/models/train_dataset/ \
    --dataset_name data/test_dataset/ \
    --data_path data/ \
    --context_path wikipedia_documents.json \
    --overwrite_cache True \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --max_answer_length 30 \
    --eval_retrieval True \
    --num_clusters 64 \
    --top_k_retrieval 20 \
    --use_faiss False \
    --retrieval_type dpr \
    --output_dir code/outputs/test_dataset/ \
    --overwrite_output_dir True \
    --do_eval False \
    --do_predict True \
    --seed 42
