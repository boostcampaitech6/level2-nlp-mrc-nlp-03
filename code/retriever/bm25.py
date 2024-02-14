from typing import List, NoReturn, Optional, Tuple, Union

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


class BM25Retrieval:
    """BM25 알고리즘을 활용해 관련된 문서를 반환합니다."""

    def __init__(
        self,
        tokenize_fn=None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        self.tokenize_fn = tokenize_fn
        self.retriever = BM25Okapi(self.contexts, self.tokenize_fn)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, docs = self.get_relevant_doc(query_or_dataset, k=topk)

            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(docs[i])

            return (doc_scores, docs)

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            _, docs_bulk = self.get_relevant_doc_bulk(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="BM25 retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(docs_bulk[idx]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        tokenized_query = self.tokenize_fn(query)
        scores = self.retriever.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        doc_scores = scores[top_indices].tolist()
        docs = [self.contexts[i] for i in top_indices]

        return doc_scores, docs

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                여러 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        doc_scores_bulk, docs_bulk = [], []
        for query in tqdm(queries, desc="Get relevant docs"):
            doc_scores, docs = self.get_relevant_doc(query, k)
            doc_scores_bulk.append(doc_scores)
            docs_bulk.append(docs)
        return doc_scores_bulk, docs_bulk
