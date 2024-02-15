#!/usr/bin/sh
# Model Arguments 설명
# --model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models

# Data Training Arguments 설명
# --dataset_name: The name of the dataset to use.
# --overwrite_cache: Overwrite the cached training and evaluation sets
# --max_seq_length: The maximum total input sequence length after tokenization.
# --pad_to_max_length: Whether to pad all samples to `max_seq_length`.
# --doc_stride: When splitting up a long document into chunks, how much stride to take between chunks.
# --max_answer_length: The maximum length of an answer that can be generated.
# --eval_retrieval: Whether to run passage retrieval using sparse embedding.
# --num_clusters: Define how many clusters to use for faiss.

# Training Arguments 설명
# --output_dir: The output directory where the model predictions and checkpoints will be written.
# --overwrite_output_dir: If True, overwrite the content of the output directory.
# --do_train: Whether to run training or not.
# --do_eval: Whether to run evaluation on the validation set or not.
# --seed: Random seed that will be set at the beginning of training.
# 추가적인 설정은 HuggingFace 참고 (HuggingFace: https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/trainer#transformers.TrainingArguments)

python3 code/train.py \
    --model_name_or_path klue/bert-base \
    --dataset_name data/train_dataset \
    --overwrite_cache True \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --max_answer_length 30 \
    --output_dir code/models/train_dataset \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --seed 42
