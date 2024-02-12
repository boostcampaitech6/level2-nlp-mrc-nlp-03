# pip install --upgrade pip
# pip install 'farm-haystack[all]'
import json
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from transformers import set_seed


def set_seed_all(seed: int = 2024):
    set_seed(seed)  # transformer seed 고정
    random.seed(seed)  # python random seed 고정
    np.random.seed(seed)  # numpy random seed 고정
    torch.manual_seed(seed)  # torch random seed 고정
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed_all(2024)

    # 위키피디아 문서 불러오기
    wiki = pd.read_json("../data/wikipedia_documents.json", orient="values")

    # 위키피디아 문서에서 텍스트만 선별
    docs = []
    for text in wiki.T["text"].drop_duplicates():
        doc = {"content": text.strip()}
        docs.append(doc)

    # Document Store
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.delete_all_documents()  # 불필요한 문서가 저장되지 않도록 인메모리를 먼저 비워줌
    document_store.write_documents(
        documents=docs, duplicate_documents="skip"
    )  # 예측할 때 참고할 문서들을 저장. 'skip': 중복된 문서는 건너뜀

    # Retriever
    retriever = BM25Retriever(document_store=document_store)  # BM25 리트리버 인스턴스

    # Reader
    # 앞서 train.py를 통해 학습한 모델의 ckpt를 불러옴
    model_ckpt = "./models/train_dataset"
    max_seq_length, doc_stride = 384, 128
    reader = FARMReader(
        model_name_or_path=model_ckpt,
        progress_bar=True,
        max_seq_len=max_seq_length,
        doc_stride=doc_stride,
        return_no_answer=False,
        use_gpu=True,
    )

    # Pipeline
    # 리트리버와 리더 과정을 연속적으로 수행
    pipe = ExtractiveQAPipeline(reader, retriever)

    # 예측할 데이터 불러오기
    test_dataset = load_from_disk("../data/test_dataset/")["validation"]

    # Inference
    predictions = pipe.run_batch(
        queries=test_dataset["question"],
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}},
    )

    # 예측값을 제출 파일 형식으로 변환
    submission = OrderedDict()
    for ex, pred in zip(test_dataset, predictions["answers"]):
        submission[ex["id"]] = pred[0].answer

    # 저장
    with open("./outputs/predictions.json", "w", encoding="utf-8") as writer:
        writer.write(json.dumps(submission, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
