import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import time
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
import os, pickle

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    HfArgumentParser,
)

from base_retrieval_class import Retrieval

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

class DenseRetrieval(Retrieval):

    def __init__(
            self, 
            args,
            tokenize_fn, 
            data_path: Optional[str] = "../data/",
            context_path: Optional[str] = "wikipedia_documents.json",) -> NoReturn:
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
            Passage 파일을 불러오고 DenseRetrieval을 선언하는 기능을 합니다.
        """

        # wikidocs.json 작업용으로 끌어오기
        super().__init__(tokenize_fn, data_path, context_path)

        self.args = args
        self.dataset = None # encoder 학습 시 사용하는 데이터셋
        self.num_neg = None # negative passage 개수 지정

        self.tokenizer = tokenize_fn # 토크나이징 함수 지정
        self.p_encoder = None # get_dense_encoder()로 생성합니다.
        self.q_encoder = None # get_dense_encoder()로 생성합니다.

        self.p_embedding = None # wikipedia.json 임베딩 파일. 없으면 get_relevant_doc_bulk로부터 받아옵니다. 
        self.indexer = None # build_faiss()로 생성합니다.

        self.prepare_valid_dataset()

    def prepare_valid_dataset(self, contexts=None, tokenizer=None):
        if contexts is None:
            contexts = self.contexts
        
        if tokenizer is None:
            tokenizer = self.tokenizer

        # valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        valid_seqs = tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

    def get_dense_encoding(self, encoder_path : Optional[str] = None, train_dataset=None, num_neg : Optional[int] = 8, model_checkpoint:Optional[str] = None) -> NoReturn:
        '''
            Summary:
                train된 모델(passage encoder과 query encoder)이 없으면 모델을 학습시키고
                모델을 pickle로 저장합니다.
                미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
            Arguments:
                encoder_path (str):
                    passage와 query의 encoder, passage embedding 등이 저장된(혹은 저장하고자 하는) 폴더의 경로를 받습니다.
                train_dataset (Union[pd.Dataframe, Dataset], Optional): 
                    학습시 사용할 데이터셋을 지정합니다. 
                num_neg (int, Optional): Defaults to 8.
                    학습 시 사용할 negative passage의 개수
                model_checkpoint (str, Optional):
                    학습 시 사용할 모델을 받습니다.
                
                encoder_path가 존재해야합니다.

            Note:
                기존 모델이 존재하는 경우, encoder_path만 입력합니다. 
                새로운 학습이 필요한 경우, encoder_path에 존재하는 encoder 삭제 후, encoder_path, train_dataset, num_neg, model_checkpoint 모두 기입이 필요합니다.
        '''

        p_ecd_name = f'trained_p_encoder.bin'
        q_ecd_name = f'trained_q_encoder.bin'
        p_embs_name = f'dense_embedding.bin'
        scores_rank_name = f'dpr_scores_rank.bin'
        
        assert encoder_path is not None, \
        'encoder의 위치를 파악하거나 encoder를 저장하기 위한 폴더 경로가 필요합니다. \
        (참고) passage와 query encoder 모두 동일한 경로로 위치해주세요'

        p_ecd_path = os.path.join(encoder_path, p_ecd_name)
        q_ecd_path = os.path.join(encoder_path, q_ecd_name)
        self.p_embs_path = os.path.join(encoder_path, p_embs_name)
        self.scores_rank_path = os.path.join(encoder_path, scores_rank_name)

        if os.path.isfile(p_ecd_path) and os.path.isfile(q_ecd_path):

            with open(p_ecd_path, 'rb') as f:
                self.p_encoder = pickle.load(f)
            with open(q_ecd_path, 'rb') as f:
                self.q_encoder = pickle.load(f)
            
            print('Passage and query encoder pickle load.')
            
            if os.path.isfile(self.p_embs_path):
                with open(self.p_embs_path, 'rb') as f:
                    self.p_embedding = pickle.load(f)
            if os.path.isfile(self.scores_rank_path):
                with open(self.scores_rank_path, 'rb') as f:
                    self.tuple_scores_rank = pickle.load(f)
        else :
            assert model_checkpoint is not None, 'encoder를 학습하기 위한 모델이 필요합니다.'
            assert train_dataset is not None, 'encoder를 학습하기 위한 데이터셋이 필요합니다.'
            assert num_neg is not None, 'encoder를 학습하기 위한 num_neg값이 필요합니다.'

            self.dataset = train_dataset
            self.num_neg = num_neg

            print('Build encoder.')
            self.prepare_in_batch_negative(dataset=self.dataset, num_neg=self.num_neg)

            self.p_encoder = BertEncoder.from_pretrained(model_checkpoint, cache_dir='../dpr_encoder/cache').to(self.args.device)
            self.q_encoder = BertEncoder.from_pretrained(model_checkpoint, cache_dir='../dpr_encoder/cache').to(self.args.device)

            self.train()

            with open(p_ecd_path, 'wb') as f:
                pickle.dump(self.p_encoder, f)
            with open(q_ecd_path, 'wb') as f:
                pickle.dump(self.q_encoder, f)
            print('Encoder trained and pickle saved.')

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

    def get_relevant_doc(
        self, query:str, k:Optional[int]=15, args=None, p_encoder=None, q_encoder=None
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
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True)
        return dot_prod_scores[0][rank][:k].tolist(), rank[0][:k].tolist()


    def get_relevant_doc_bulk( # k는 20이상이면 그냥 reader 개선하는게 좋음
        self, queries: List, k: Optional[int]=20, args=None, p_encoder=None, q_encoder=None
    ) -> Tuple[List, List]:
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        # 메모리 사용량 프로파일링 시작
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()  # 캐시된 메모리 정리

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
            q_dataset = TensorDataset(
                q_seqs_val['input_ids'], q_seqs_val['attention_mask'], q_seqs_val['token_type_ids']
            )
            q_dataloader = DataLoader(q_dataset, batch_size = self.args.per_device_eval_batch_size)

            q_embs = []
            
            # for batch in q_dataloader:
            for batch in tqdm(q_dataloader, desc='query encoding: '):
        
                batch = tuple(t.to(args.device) for t in batch)
                q_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                q_emb = q_encoder(**q_inputs).to('cpu')
                # q_emb = q_encoder(**p_inputs).to(args.device)
                q_embs.append(q_emb)

                # 메모리 사용량 확인
                # print(f"Peak memory usage by tensors: {torch.cuda.max_memory_allocated() / 1e6} MB")

            q_embs = torch.stack(q_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

            p_embs = None
            if self.p_embedding is None:
                p_embs = []
                # for batch in self.passage_dataloader:
                for batch in tqdm(self.passage_dataloader, desc='wiki passage encoding: '):

                    batch = tuple(t.to(args.device) for t in batch)
                    p_inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                    }
                    p_emb = p_encoder(**p_inputs).to('cpu')
                    # p_emb = p_encoder(**p_inputs).to(args.device)
                    p_embs.append(p_emb)

                p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)
                with open(self.p_embs_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)

            else:
                p_embs = self.p_embedding

        dot_prod_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True)
        d, r = [], []

        for i in tqdm(range(rank.shape[0]), desc=': '):
            d.append(dot_prod_scores[i][rank][:k].tolist())
            r.append(rank[i][:k].tolist())
      
        return d, r
    

def main():

    # GPU OOM : https://modernflow.tistory.com/88
    # PYTORCH_CUDA_ALLOC_CONF 환경 변수 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    torch.cuda.empty_cache()
    
    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    train_dataset = load_dataset("squad_kor_v1")["train"]


    # # 메모리가 부족한 경우 일부만 사용하세요 !
    num_sample = 20000
    sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    train_dataset = train_dataset[sample_idx]

    args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir="dense_retrieval",
        learning_rate=3e-4,
        per_device_train_batch_size=4,  # 8이면 top 29 31, 32이면 top accuracy 
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01
    )
    model_checkpoint = "klue/bert-base"

    # # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    tokenize_fn = AutoTokenizer.from_pretrained(model_checkpoint)
    # p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    # q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    # Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
    retriever = DenseRetrieval(
        args=args,
        tokenize_fn=tokenize_fn,
        data_path="../../data",
        context_path="wikipedia_documents.json"
    )

    retriever.get_dense_encoding(
        encoder_path='../dense_retrieval', 
        train_dataset=train_dataset, 
        num_neg=8, 
        model_checkpoint=model_checkpoint
    )


    org_dataset = load_from_disk('../../data/train_dataset')

    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    
    df = retriever.retrieve(query_or_dataset=full_ds, topk=15)
    print(df.head())
    df.to_c



    
if __name__ == '__main__':
    main()


