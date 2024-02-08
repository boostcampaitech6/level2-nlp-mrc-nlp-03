#!/usr/bin/sh
python3 code/train.py \
    --model_name_or_path monologg/koelectra-small-v3-finetuned-korquad \
    --output_dir code/models/train_dataset \
    --do_train \
    --do_eval
