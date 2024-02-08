#!/usr/bin/sh
python3 code/inference.py \
    --output_dir code/outputs/test_dataset/ \
    --dataset_name data/test_dataset/ \
    --model_name_or_path code/models/train_dataset/ \
    --do_predict
