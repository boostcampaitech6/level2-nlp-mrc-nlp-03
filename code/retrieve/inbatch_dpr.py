import json
import random
from pprint import pprint
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
import os, pickle

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
    
class DenseRetrieval:
    def __init__(self, args, dataset, model_checkpoint):
        """
            학습과 추론에 필요한 객체들을 받아서 속성으로 저장합니다.
            객체가 instantiate될 때 in-batch negative가 생긴 데이터를 만들도록 함수를 수행합니다.
        """
        self.set_seed(2023)
        self.args = args
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.prepare_in_batch_negative()
        self.q_encoder, self.p_encoder = self.prepare_encoders()

    def set_seed(self, seed):  
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def prepare_encoders(self, encoder_path="../models/dpr"):
        """
        Summary:
            q_encoder와 p_encoder를 반환하는 함수입니다.
            만약 이미 학습된 encoder가 있다면 불러오고, 없다면 새로 만듭니다.
        """
        q_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_q_encoder")
        p_encoder_path = os.path.join(encoder_path, f"{self.model_checkpoint}_p_encoder")

        if os.path.exists(q_encoder_path) and os.path.exists(p_encoder_path):
            q_encoder = BertEncoder.from_pretrained(q_encoder_path)
            p_encoder = BertEncoder.from_pretrained(p_encoder_path)
            print("Encoder is loaded.")
        else:
            print("Encoder is not found. Build new encoder.")
            p_encoder = BertEncoder.from_pretrained(self.model_checkpoint)
            q_encoder = BertEncoder.from_pretrained(self.model_checkpoint)

            if torch.cuda.is_available():
                p_encoder.to('cuda')
                q_encoder.to('cuda')

            p_encoder.save_pretrained(q_encoder_path)
            q_encoder.save_pretrained(p_encoder_path)

            p_encoder, q_encoder = self.train(p_encoder, q_encoder, self.args)

        return q_encoder, p_encoder 

    def load_wiki_contexts(self, path="../../data/wikipedia_documents.json"):
        """
        Summary:
            위키피디아의 context를 불러옵니다.
        """
        with open(path, "r") as f:
            wiki = json.load(f)

        contexts = list(set([wiki[doc]["text"] for doc in wiki]))
        return contexts

    def get_wiki_dense_embedding(self):
        """
        Summary:
            Dense Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 pickle을 불러옵니다.
        """
        pickle_name = "dense_embedding.bin"
        data_path = "../../data"
        emd_path = os.path.join(data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as f:
                wiki_embs = pickle.load(f)
            print("Embedding pickle load.")
        else:
            print("Build passage embdding")
            wiki_contexts = self.load_wiki_contexts()
            wiki_embs = self.embedding_passage(self.tokenizer, wiki_contexts)
            with open(emd_path, "wb") as f:
                pickle.dump(wiki_embs, f)

        return wiki_embs

    def embedding_passage(self, tokenizer, contexts):
        """
        Summary:
            Passage를 받아서 embedding을 반환합니다.
        """
        if torch.cuda.is_available():
            self.p_encoder.to('cuda')

        with torch.no_grad():
            self.p_encoder.eval()
            p_embs = []

            for p in tqdm(contexts):
                p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                p_emb = self.p_encoder(**p).to('cpu').numpy()
                p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()
            return p_embs

    def prepare_in_batch_negative(self, dataset=None, tokenizer=None):
        """Huggingface의 Dataset을 받아오면, in-batch negative를 추가해서 Dataloader를 만들어줍니다."""
        if dataset is None:
            # train_dataset = self.dataset['train']
            train_dataset = self.dataset['train'][:128]
            valid_dataset = self.dataset['validation']

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        train_corpus = np.array(list({example for example in train_dataset["context"]}))

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

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        # 3. valid dataset 만들어주기
        valid_seqs = tokenizer(valid_dataset["context"], padding="max_length", truncation=True, return_tensors="pt")
        valid_dataset = TensorDataset(
            valid_seqs["input_ids"], 
            valid_seqs["attention_mask"], 
            valid_seqs["token_type_ids"]
        )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.per_device_train_batch_size)

    def train(self, p_encoder, q_encoder, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if not any(no_d in n for no_d in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
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

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        # train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in train_iterator:
        for i in range(int(args.num_train_epochs)):
            print(f"Epoch {i+1}/{args.num_train_epochs}")
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

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

                    p_outputs = p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

                    # target: position of positive samples = diagonal element
                    targets = torch.arange(0, args.per_device_train_batch_size).long()
                    if torch.cuda.is_available():
                        targets = targets.to('cuda')

                    sim_scores = F.log_softmax(sim_scores, dim=1)
                
                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    p_encoder.zero_grad()
                    q_encoder.zero_grad()
                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        return p_encoder, q_encoder
    
    def get_relevant_doc_from_wiki(self, query, k=1):
        """
        Summary:
            위키피디아의 context를 받아서 유사한 passage를 반환합니다.
        """
        wiki_embs = self.get_wiki_dense_embedding()
        wiki_contexts = self.load_wiki_contexts()

        scores, indices = self.get_relevant_doc(query, k, wiki_embs)

        print(f"Query: {query}")
        for i in range(k):
            print(f"Top-{i+1} passage with score {scores.squeeze()[indices[i]]}")
            print(wiki_contexts[indices[i]])

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None, p_embs=None):
        """Args query (str) 문자열로 주어진 질문입니다.

        k (int)
            상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.

        args
            Configuration을 필요한 경우 넣어줍니다.
            만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Do
        --
        1. query를 받아서 embedding을 하고
        2. 전체 passage와의 유사도를 구한 후
        3. 상위 k개의 문서 index를 반환합니다.
        """
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        if p_embs is None:
            p_embs = self.get_wiki_dense_embedding()

        with torch.no_grad():
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        return dot_prod_scores, rank[:k]


def main():
    dataset = load_dataset("squad_kor_v1")

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01
    )

    retriever = DenseRetrieval(
        args=args,
        dataset=dataset,
        model_checkpoint="bert-base-multilingual-cased",
    )

    retriever.get_relevant_doc_from_wiki("대한민국의 수도는 어디인가요?", k=5)

if __name__ == "__main__":
    main()
