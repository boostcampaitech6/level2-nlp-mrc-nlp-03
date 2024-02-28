import argparse
from datasets import load_from_disk, DatasetDict
from ragatouille import RAGPretrainedModel
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사전학습된 모델을 사용해 쿼리와 관련있는 문서를 검색합니다.")
    parser.add_argument('--data_path', type=str, default='../data/test_dataset/',
                        help='데이터셋 저장 경로')
    parser.add_argument('--index_path', type=str, default='.ragatouille/colbert/indexs/index/',
                        help='인덱스 작업을 마친 모델이 저장된 경로')
    parser.add_argument('--topk', type=int, default=5,
                        help='검색할 문서의 개수')
    parser.add_argument('--output_dir', type=str, default='../data/colbert_dataset',
                        help='검색 작업을 마친 데이터셋을 저장할 경로')
    return parser.parse_args()

def search(args: argparse.Namespace) -> None:
    """사전학습된 모델을 사용해 쿼리와 관련있는 문서를 검색합니다.

    Args:
        args (argparse.Namespace): CLI에서 입력받은 검색 작업에 필요한 파일 경로와 저장 경로
    """
    # 쿼리 데이터셋을 불러옵니다.
    dataset = load_from_disk(args.data_path)['validation']
    
    # 사전에 인덱싱 작업을 마친 모델을 불러옵니다.
    RAG = RAGPretrainedModel.from_index(
        index_path=args.index_path        
    )

    def get_relevant_docs(example: Dict[str, Any]) -> Dict[str, str]:
        """ColBERT 모델을 사용해 쿼리에 관련된 문서를 검색합니다. 

        Args:
            example (Dict[str, Any]): 'question'을 포함한 Dataset의 샘플

        Returns:
            Dict[str, str]: 쿼리에 관련된 문서를 합한 딕셔너리
        """
        query = example['question']
        results = [result['content'] for result in RAG.search(query=query, k=args.topk)]
        context = " ".join(results)
        return {'context': context}
    
    # 모든 쿼리에 대해 관련된 문서를 찾아옵니다.
    dataset = dataset.map(get_relevant_docs)

    # 검색 작업을 마친 데이터셋을 저장합니다.
    output = DatasetDict({'validation': dataset})
    output.save_to_disk(args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    search(args)