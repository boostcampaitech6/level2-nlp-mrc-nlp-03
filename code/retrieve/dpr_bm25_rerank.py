import os, json
from contextlib import contextmanager
import torch
from tqdm.auto import tqdm
import time
import pandas as pd
import numpy as np

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BM25_DPRpipeline:
    def __init__(self, bm25, dpr):
        self.bm25 = bm25
        self.dpr = dpr

    def get_sparse_embedding(self):
        self.dpr.get_sparse_embedding()
        self.bm25.get_sparse_embedding()

    def retrieve(self, query_or_dataset, topk=5):
        # return self.dpr.retrieve(query_or_dataset, topk)
        total = []
        initial_k = 2000
        with timer("query exhaustive search"):
            dpr_doc_scores, dpr_incidies = self.dpr.get_relevant_doc_bulk(
                query_or_dataset["question"], k=initial_k
            )
            bm25_doc_scores, bm25_incidies = self.bm25.get_relevant_doc_bulk(
                query_or_dataset["question"], k=initial_k
            )

        lambda_value = 1.1
        sum_scores = []
        for i in range(len(dpr_doc_scores)):
            sum_score = [0 for _ in range(len(self.bm25.contexts))]
            for idx, j in enumerate(dpr_incidies[i]):
                sum_score[j] += dpr_doc_scores[i][idx] * lambda_value
            for idx, j in enumerate(bm25_incidies[i]):
                sum_score[j] += bm25_doc_scores[i][idx]
            sum_scores.append(sum_score)
        sum_scores = np.array(sum_scores)

        new_doc_scores = []
        new_doc_incidies = []  
        for i in range(len(dpr_doc_scores)):
            sorted_result = np.argsort(sum_scores[i, :])[::-1]
            new_doc_scores.append(sum_scores[i, sorted_result].tolist()[:topk])
            new_doc_incidies.append(sorted_result.tolist()[:topk])
        
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": " ".join([self.bm25.contexts[pid] for pid in new_doc_incidies[idx]]),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
            
        cqas = pd.DataFrame(total)
        return cqas