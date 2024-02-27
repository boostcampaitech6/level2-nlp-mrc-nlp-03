# How To Use ColBERT

## 0. Requirements
```bash
pip install ragatouille -q
pip uninstall --y faiss-cpu -q
pip install faiss-gpu -q
```

## 1. Train
ColBERT의 쿼리-문서 매칭 기능을 사전학습 시킵니다.
```bash
python3 train.py
```

## 2. Indexing
사전학습된 ColBERT를 활용하여 문서 검색 속도를 높이기 위해 검색 대상 문서에 대한 인덱싱을 수행합니다.
```bash
python3 indexing.py
```

## 3. Searching
모든 쿼리에 대한 관련 문서를 검색합니다.
```bash
python3 searching.py
```
## 4. Inference
쿼리와 검색 결과를 활용하여 답변을 추론합니다.
```bash
python3 inference.py
```
---
**NOTE:** FAISS-GPU를 사용하는 작업(Indexing & Searching)은 CUDA 드라이버를 필요로 합니다. 현재 AISTAGES 서버 환경에는 드라이버 설치가 어려워 Colab 또는 CUDA 드라이버 설치가 가능한 외부 환경에서 작업하는 것을 권장합니다.