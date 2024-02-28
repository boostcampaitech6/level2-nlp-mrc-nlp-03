from typing import List, NoReturn, Optional, Tuple, Union

import json, os, pickle
import random
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time

from contextlib import contextmanager
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from .base_retrieval_class import Retrieval
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output
    
class DenseRetrieval(Retrieval):
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
        self.model_checkpoint = "bert-base-multilingual-cased"
        self.precision=16,
        self.args = TrainingArguments(
                output_dir="dense_retireval",
                evaluation_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=3,
                weight_decay=0.01,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.scaler = torch.cuda.amp.GradScaler()
        self.prepare_encoders()

    def prepare_encoders(self, encoder_path="./models/dpr"):
        """
        Summary:
            q_encoder와 p_encoder를 반환하는 함수입니다.
            만약 이미 학습된 encoder가 있다면 불러오고, 없다면 새로 만듭니다.
        """
        q_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_{self.precision}_q_encoder")
        p_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_{self.precision}_p_encoder")

        if os.path.exists(q_encoder_path) and os.path.exists(p_encoder_path):
            self.q_encoder = BertEncoder.from_pretrained(q_encoder_path)
            self.p_encoder = BertEncoder.from_pretrained(p_encoder_path)
            print("Encoder Path : ", q_encoder_path, p_encoder_path)

            if torch.cuda.is_available():
                self.q_encoder.to('cuda')
                self.p_encoder.to('cuda')
            print("Encoder is loaded.")
        else:
            self.train()
            print("Encoder is trained.")
            self.save_encoder()
            print("Encoder is saved.")
        
    def get_sparse_embedding(self):
        with timer("Dense Passage Embedding"):
            # Pickle을 저장합니다.
            pickle_name = f"{self.model_checkpoint.replace('/', '-')}_{self.precision}_dense_embedding.bin"
            emb_path = os.path.join(self.data_path, pickle_name)

            if os.path.isfile(emb_path):
                with open(emb_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                print("Dense Embedding load.")
            else:
                self.p_embedding = self.embedding_passage()
                with open(emb_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                print("Dense Embedding saved.")
        self.p_embedding = torch.Tensor(self.p_embedding)

    def embedding_passage(self):
        """
        Summary:
            Passage를 받아서 embedding을 반환합니다.
        """
        if torch.cuda.is_available():
            self.p_encoder.to('cuda')

        with torch.no_grad():
            self.p_encoder.eval()
            p_embs = []

            for p in tqdm(self.contexts):
                p = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                p_emb = self.p_encoder(**p).to('cpu').numpy()
                p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()
            return p_embs
        
    def embedding_queries(self, queries):
        """
        Summary:
            Query를 받아서 embedding을 반환합니다.
        """
        if torch.cuda.is_available():
            self.q_encoder.to('cuda')

        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = []

            for q in tqdm(queries):
                q = self.tokenizer(q, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_emb = self.q_encoder(**q).to('cpu').numpy()
                q_embs.append(q_emb)
            
            q_embs = torch.Tensor(q_embs).squeeze()
            return q_embs
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            q_embedding = self.embedding_queries([query])

        with timer("query ex search"):
            result = torch.matmul(q_embedding, torch.transpose(self.p_embedding, 0, 1))
            sorted_result = torch.argsort(result, dim=1, descending=True).squeeze()

        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # Pickle을 저장합니다.
        dpr_ranking = f"{self.model_checkpoint.replace('/', '-')}_{self.precision}_dpr_ranking.bin"
        dpr_scores = f"{self.model_checkpoint.replace('/', '-')}_{self.precision}_dpr_scores.bin"
        dpr_ranking_path = os.path.join(self.data_path, dpr_ranking)
        dpr_scores_path = os.path.join(self.data_path, dpr_scores)

        if os.path.isfile(dpr_ranking_path) and os.path.isfile(dpr_scores_path):
            with open(dpr_ranking_path, "rb") as file:
                doc_indices = pickle.load(file)
            with open(dpr_scores_path, "rb") as file:
                doc_scores = pickle.load(file)
        else:
            with timer("transform"):
                q_embedding = self.embedding_queries(queries)

            with timer("query ex search"):
                print(type(q_embedding), type(self.p_embedding))
                result = torch.matmul(q_embedding, torch.transpose(self.p_embedding, 0, 1))
                
                doc_scores = []
                doc_indices = []
                for i in range(result.shape[0]):
                    sorted_result = torch.argsort(result[i, :], dim=0, descending=True)
                    doc_scores.append(result[i, :][sorted_result].tolist())
                    doc_indices.append(sorted_result.tolist())

            with open(dpr_ranking_path, "wb") as file:
                pickle.dump(doc_indices, file)
            with open(dpr_scores_path, "wb") as file:
                pickle.dump(doc_scores, file)
            
        doc_scores = [doc_score[:k] for doc_score in doc_scores]
        doc_indices = [doc_index[:k] for doc_index in doc_indices]
        return doc_scores, doc_indices
    
    def prepare_in_batch_negative(self, dataset=None, tokenizer=None):
        """Huggingface의 Dataset을 받아오면, in-batch negative를 추가해서 Dataloader를 만들어줍니다."""
        if dataset is None:
            self.dataset = load_dataset("squad_kor_v1")
            dataset = self.dataset
        # train_dataset = dataset['train']
        train_dataset = dataset['train'][:20000]
        valid_dataset = dataset['validation']
        print("Train dataset is loaded. Train dataset length: ", len(train_dataset))
        print("Validation dataset is loaded. Validation dataset length: ", len(valid_dataset))

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(train_dataset["question"], padding="max_length", truncation=True, return_tensors="pt")
        p_seqs = tokenizer(train_dataset['context'], padding="max_length", truncation=True, return_tensors="pt")

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        batch_size = self.args.per_device_train_batch_size
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    def train(self, args=None):
        if args is None:
            args = self.args

        self.prepare_in_batch_negative()
        self.p_encoder = BertEncoder.from_pretrained(self.model_checkpoint)
        self.q_encoder = BertEncoder.from_pretrained(self.model_checkpoint)

        if torch.cuda.is_available():
            self.p_encoder.to('cuda')
            self.q_encoder.to('cuda')

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        t_total = (
            len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        # train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in train_iterator:
        for i in range(int(args.num_train_epochs)):
            print(f"Epoch {i+1}/{args.num_train_epochs}")
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for step, batch in enumerate(tepoch):
                    self.p_encoder.train()
                    self.q_encoder.train()

                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)

                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2]
                    }

                    q_inputs = {
                        "input_ids": batch[3],
                        "attention_mask": batch[4],
                        "token_type_ids": batch[5],
                    }

                    # target: position of positive samples = diagonal element
                    targets = torch.arange(0, args.per_device_train_batch_size).long()
                    if torch.cuda.is_available():
                        targets = targets.to('cuda')

                    with torch.cuda.amp.autocast():
                        p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                        q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                        # Calculate similarity score & loss
                        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                        sim_scores = F.log_softmax(sim_scores, dim=1)
                    
                        loss = F.nll_loss(sim_scores, targets)

                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    scheduler.step()
                    self.scaler.update()
                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()
                    global_step += 1
                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

                    if step % 100 == 0:
                        print(f"Step {step} : Loss {loss.item()}")

    def evaluate(self, k=5, dataset=None, args=None):
        if dataset == None:
            valid_dataset = load_dataset("squad_kor_v1")['validation']
            valid_corpus = list(set(valid_dataset["context"]))
        else:
            valid_dataset = dataset


        if torch.cuda.is_available():
            self.q_encoder.to('cuda')
            self.p_encoder.to('cuda')
        self.p_encoder.eval()
        self.q_encoder.eval()
        
        eval_embedding_path = os.path.join(self.data_path, "eval_embedding.bin")

        with torch.no_grad():
            if os.path.isfile(eval_embedding_path) and False:
                with open(eval_embedding_path, "rb") as file:
                    p_embs = pickle.load(file)
            else:
                p_embs = []
                for p in tqdm(valid_corpus, desc="Valid Embedding Passage"):
                    p_inputs = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    p_outputs = self.p_encoder(**p_inputs).to('cpu').numpy()
                    p_embs.append(p_outputs)
                
                with open(eval_embedding_path, "wb") as file:
                    pickle.dump(p_embs, file)
            p_embs = torch.Tensor(p_embs).squeeze()

            # 2. valid_query embedding
            correct = 0
            for i in tqdm(range(len(valid_dataset))):
                ground_truth = valid_dataset[i]['context']
                query = valid_dataset[i]['question']

                if not ground_truth in valid_corpus:
                    valid_corpus.append(ground_truth)

                q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_emb = self.q_encoder(**q_seqs_val).to('cpu')

                dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

                if ground_truth in [valid_corpus[rank[i]] for i in range(k)]:
                    correct += 1

            print(f"Accuracy : {correct/len(valid_dataset)}")
    
    def save_encoder(self, encoder_path="../models/dpr"):
        q_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_q_encoder")
        p_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_p_encoder")

        self.q_encoder.save_pretrained(q_encoder_path)
        self.p_encoder.save_pretrained(p_encoder_path)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    dpr = DenseRetrieval(tokenizer.tokenize, "../data/", "wikipedia_documents.json")
    dpr.prepare_encoders()
    dpr.prepare_in_batch_negative()
    dpr.evaluate(k=20)