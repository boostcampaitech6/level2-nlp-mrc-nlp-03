import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import time
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from datasets import Dataset
import os, pickle

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

set_seed(42) # magic number :)

print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:

    # def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):
    def __init__(self, args, dataset, tokenizer, num_neg : Optional[int] = 2) -> NoReturn:

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = None # get_dense_encoder()로 생성합니다.
        self.q_encoder = None # get_dense_encoder()로 생성합니다.

        self.indexer = None # build_faiss()로 생성합니다.

        self.prepare_in_batch_negative(num_neg=self.num_neg)

    def prepare_in_batch_negative(self, dataset=None, num_neg = None, tokenizer=None) -> NoReturn:

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        if num_neg is None:
            num_neg = self.num_neg

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:

            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


    def get_dense_encoding(self, encoder_path : Optional[str] = None, model_checkpoint:Optional[str] = None) -> NoReturn:
        '''
          Summary:
              train된 모델(passage encoder과 query encoder)이 없으면 모델을 학습시키고
              모델을 pickle로 저장합니다.
              미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.

          Arguments:
              encoder_path (Optional[str]):
                  passage와 query의 encoder가 저장된(혹은 저장하고자 하는) 폴더의 경로를 받습니다.
              model_checkpoint :
                  미리 학습된 모델이 없을 경우, 학습에 사용하려는 모델을 받습니다.
        '''

        p_ecd_name = f'trained_p_encoder.bin'
        q_ecd_name = f'trained_q_encoder.bin'

        assert encoder_path is not None, \
        'encoder의 위치를 파악하거나 encoder를 저장하기 위한 폴더 경로가 필요합니다. \
        (참고) passage와 query encoder 모두 동일한 경로로 위치해주세요'


        p_ecd_path = os.path.join(encoder_path, p_ecd_name)
        q_ecd_path = os.path.join(encoder_path, q_ecd_name)


        if os.path.isfile(p_ecd_path) and os.path.isfile(q_ecd_path):
            with open(p_ecd_path, 'rb') as f:
                self.p_encoder = pickle.load(f)
            with open(q_ecd_path, 'rb') as f:
                self.q_encoder = pickle.load(f)
            print('Passage and query encoder pickle load.')
        else :
            assert model_checkpoint is not None, \
            'encoder를 학습하기 위한 모델이 필요합니다.'

            print('Build encoder.')
            self.p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(self.args.device)
            self.q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(self.args.device)

            self.train()

            with open(p_ecd_path, 'wb') as f:
                pickle.dump(self.p_encoder, f)
            with open(q_ecd_path, 'wb') as f:
                pickle.dump(self.q_encoder, f)
            print('Encoder trained and pickle saved.')

    def train(self, args=None) -> NoReturn:

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple(List, List), pd.DataFrame]:
        '''
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relavant_doc`을 통해 유사도를 구합니다.
                Datset 형태는 query를 포함한 HF.Datset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query  (train/valid)  -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query  (test)         -> Retriever한 Passage만 반환합니다.
        '''

        assert self.p_encoder is not None and self.q_encoder is not None, "get_dense_encoding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_prod_score, rank = self.get_relevant_doc(query_or_dataset, k=topk)
            print(f"[Search Query] {query_or_dataset}\n")

            for i, idx in enumerate(rank):
                print(f"Top-{i + 1}th Passage (Index {idx})")
                pprint(self.dataset['context'][idx])

            return (doc_prod_scores, [self.dataset['context'][pid] for pid in range(rank)])
        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer('query exhastive search'):
                doc_prod_scores, ranks = self.get_relevant_doc_bulk(
                    query_or_dataset['question'], k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc='Dense retrieval: ')
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    'question': example['question'],
                    'id' : example['id'],

                    # Retrieve한 Passage의 id, context를 반환합니다.
                    'context_id': ranks[idx],
                    'context': ''.join([self.dataset['context'][pid] for pid in rank[idx]])
                }

                if 'context' in example.keys() and 'answers' in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp['origin_context'] = example['context']
                    tmp['answers'] = example['answers']

                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(
        self, query:str, k:Optional[int]=1, args=None, p_encoder=None, q_encoder=None
      ) -> Tuple[List, List]:

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            
            for batch in self.passage_dataloader: 

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return dot_prod_scores[rank][:k].tolist(), rank[:k].tolist()


    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int]=1, args=None, p_encoder=None, q_encoder=None
    ) -> Tuple[List, List]:
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            # 이부분이 뭔가 indexer들어가는 부분같은디..
            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        d, r = [], []
        for i in range(rank.shape[0]):
            d.append(dot_prod_scores[i][rank][:k].tolist())
            r.append(rank[i][:k].tolist())
      
        return d, r
    



class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()


    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
        ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output
