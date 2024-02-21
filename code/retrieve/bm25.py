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
from datasets import Dataset, concatenate_datasets, load_from_disk
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from .base_retrieval_class import Retrieval


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25(Retrieval):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        super().__init__(tokenize_fn, data_path, context_path)
        self.tokenize_fn = tokenize_fn
        self.bm25 = None

    def get_sparse_embedding(self):
        with timer("BM25 Embedding"):
            # Pickle을 저장합니다.
            bm25v_pickle = "bm25v.bin"
            bm25v_path = os.path.join(self.data_path, bm25v_pickle)

            if os.path.isfile(bm25v_path):
                with open(bm25v_path, "rb") as file:
                    self.bm25 = pickle.load(file)
                print("BM25 pickle load.")
            else:
                self.bm25 = BM25Okapi(self.contexts, tokenizer=self.tokenize_fn)
                with open(bm25v_path, "wb") as file:
                    pickle.dump(self.bm25, file)
                print("BM25 pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("query 토큰화"):
            tokenized_query = self.tokenize_fn(query)

        with timer("문서 search"):
            result = self.bm25.get_scores(tokenized_query)

        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("query 토큰화"):
            tokenized_queris = [self.tokenize_fn(query) for query in queries]

        with timer("문서 search"):
            
            # Pickle을 저장합니다.
            bm25_ranking = "bm25_ranking.bin"
            bm25_scores = "bm25_scores.bin"
            bm25_ranking_path = os.path.join(self.data_path, bm25_ranking)
            bm25_scores_path = os.path.join(self.data_path, bm25_scores)

            if os.path.isfile(bm25_ranking_path) and os.path.isfile(bm25_scores_path):
                with open(bm25_ranking_path, "rb") as file:
                    doc_indices = pickle.load(file)
                with open(bm25_scores_path, "rb") as file:
                    doc_scores = pickle.load(file)
            else:
                result = np.array(
                    [
                        self.bm25.get_scores(tokenized_query)
                        for tokenized_query in tqdm(tokenized_queris, desc="관련 문서 수집: ")
                    ]
                )

                doc_scores = []
                doc_indices = []

                for i in range(result.shape[0]):
                    sorted_result = np.argsort(result[i, :])[::-1]
                    doc_scores.append(result[i, :][sorted_result].tolist())
                    doc_indices.append(sorted_result.tolist())

                with open(bm25_ranking_path, "wb") as file:
                    pickle.dump(doc_indices, file)
                with open(bm25_scores_path, "wb") as file:
                    pickle.dump(doc_scores, file)
        
        doc_scores = [doc_score[:k] for doc_score in doc_scores]
        doc_indices = [doc_index[:k] for doc_index in doc_indices]
        return doc_scores, doc_indices
