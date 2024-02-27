import json
import argparse
from ragatouille import RAGPretrainedModel
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColBERT를 사용해 위키피디아 문서에 인덱싱 작업을 수행합니다.")
    parser.add_argument('--wikipedia_path', type=str, default='../data/wikipedia_documents.json',
                        help='Wikipedia documents JSON 경로')
    parser.add_argument('--model_name_or_path', type=str, default='.ragatouille/colbert/checkpoint/',
                        help='사전 학습 모델 또는 경로')
    parser.add_argument('--index_root', type=str, default=None,
                        help='구축된 인덱스를 저장할 경로')
    parser.add_argument('--index_name', type=str, default='index',
                        help='생성할 인덱스 파일의 이름')
    return parser.parse_args()

def index(args: argparse.Namespace) -> None:
    """사전학습된 ColBERT를 사용해 위키피디아 코퍼스에 Index를 부여합니다.

    Args:
        args (argparse.Namespace): CLI에서 입력받은 파일 경로와 모델 정보
    """
    # 위키피디아 코퍼스를 불러옵니다.
    with open(args.wikipedia_path, encoding='utf-8') as f:
        wiki = json.load(f)
    corpus = list({v["text"]: None for v in wiki.values()}.keys())

    # 사전 학습된 ColBERT 모델을 불러옵니다.
    RAG = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        index_root=args.index_root
    )
    
    # 인덱싱 작업을 수행합니다.
    RAG.index(
        collection=corpus,
        index_name=args.index_name,
        max_document_length=384,
        split_documents=True
    )

if __name__ == '__main__':
    args = parse_args()
    index(args)