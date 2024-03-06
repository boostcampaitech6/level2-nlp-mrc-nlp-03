import json
import argparse
from typing import List, Dict, Union
from datasets import DatasetDict, load_from_disk
from ragatouille import RAGTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="주어진 데이터셋과 위키피디아 문서를 사용해 ColBERT를 학습시킵니다.")
    parser.add_argument('--train_path', type=str, default='../data/train_dataset/', help='train datset 경로')
    parser.add_argument('--wikipedia_path', type=str, default='../data/wikipedia_documents.json', help='Wikipedia documents JSON 경로')
    parser.add_argument('--model_name_or_path', type=str, default='klue/bert-base', help='사전 학습 모델 또는 경로')
    return parser.parse_args()

def train(args: argparse.Namespace) -> None:
    """학습 데이터셋과 위키피디아 문서를 사용해 ColBERT를 학습시킵니다. 

    Args:
        args (argparse.Namespace): CLI에서 입력받은 학습 관련 파라미터 및 파일 경로 
    """
    # 학습 데이터를 불러옵니다.
    dataset: DatasetDict = load_from_disk(args.train_path)['train']
    
    # 중복 문서를 제거하며 위키피디아 코퍼스를 불러옵니다.
    with open(args.wikipedia_path, encoding='utf-8') as f:
        wiki: Dict[str, Dict[str, str]] = json.load(f)
    corpus: List[str] = list({v["text"]: None for v in wiki.values()}.keys())

    # ColBERT 학습을 위한 triple을 마련합니다. 
    triples: List[List[Union[str, int]]] = [[ex['question'], ex['context'], 1] for ex in dataset]

    # Trainer 객체를 생성합니다.
    trainer = RAGTrainer(model_name=args.model_name_or_path, pretrained_model_name=args.model_name_or_path, language_code="ko")

    # 쿼리와 문서 간의 매칭 성능을 높이기 위해 Hard Negative Passage를 마련합니다.
    trainer.prepare_training_data(
        raw_data=triples,
        all_documents=corpus,
        mine_hard_negatives=True,
        hard_negative_model_size="large",
        pairs_with_labels=True,
    )

    # 학습을 시작합니다.
    trainer.train(
        batch_size=16,
        nbits=4,
        maxsteps=500000,
        use_ib_negatives=True,
        dim=128,
        learning_rate=5e-6,
        doc_maxlen=384,
        use_relu=False,
        warmup_steps="auto",
    )

if __name__ == '__main__':
    args = parse_args()
    train(args)