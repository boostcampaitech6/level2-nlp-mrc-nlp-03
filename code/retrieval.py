from typing import List, NoReturn, Optional, Tuple, Union

import json
import os
import pickle
import random
import time
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from retrieve.base_retrieval_class import Retrieval
from retrieve.bm25 import BM25
from retrieve.tf_idf import TfidfRetrieval
from retrieve.dpr_editing import BertEncoder, DenseRetrieval

from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, help="")
    parser.add_argument(
        "--model_name_or_path",
        # metavar="bert-base-multilingual-cased",
        metavar="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, help="")
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    parser.add_argument("--bm25", default='dpr', type=bool, 
                        help="BM25는 'bm25', TF-IDF는 'tf_idf', DPR은 'dpr'")
    
    parser.add_argument("--dpr_encoder_path", metavar="./dense_retireval", type=str, 
                        help="dpr encoder 경로 지정 시 입력")
    parser.add_argument("--dpr_model_checkpoint", metavar="klue/bert-base", type=str, help="dpr 학습 모델 입력")
    parser.add_argument("--dpr_train_num", metavar=20000, type=int, help="dpr 학습시 train data 개수")

    args = parser.parse_args()

    # Test sparse
    # org_dataset = load_from_disk(args.dataset_name)
    org_dataset = load_from_disk('../data/train_dataset')
    org_dataset['train'] = org_dataset['train'].remove_columns('document_id')
    org_dataset['validation'] = org_dataset['validation'].remove_columns('document_id')
    org_dataset['train'] = org_dataset['train'].remove_columns('__index_level_0__')
    org_dataset['validation'] = org_dataset['validation'].remove_columns('__index_level_0__')

    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트

    
        
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    if args.rt_type == 'bm25':
        retriever = BM25(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
    elif args.rt_type == 'tf_idf':
        retriever = TfidfRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
    elif args.rt_type == 'dpr':
        dpr_args = TrainingArguments(
            learning_rate=3e-4,
            per_device_train_batch_size=8,  # 8이면 top acc 29 ~ 31, 32이면 top acc
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01
        )
        retriever = DenseRetrieval(
            args= dpr_args,
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
    if args.bm25 != 'dpr':
        retriever.get_sparse_embedding()
    else :
        train_dataset = load_dataset("squad_kor_v1")["train"]
        num_sample = args.dpr_train_num
        sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
        train_dataset = train_dataset[sample_idx]

        retriever.get_dense_encoding(
            encoder_path=args.dpr_encoder_path,
            train_dataset=train_dataset,
            num_neg=8,
            model_checkpoint=args.dpr_model_checkpoint
        )

    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
