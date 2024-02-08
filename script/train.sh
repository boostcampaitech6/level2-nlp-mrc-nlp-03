#!/usr/bin/sh
python3 code/train.py \
    --model_name_or_path monologg/koelectra-base-v3-finetuned-korquad \
    --output_dir code/models/train_dataset \
    --overwrite_cache \
    --overwrite_output_dir \
    --do_train \
    --do_eval
