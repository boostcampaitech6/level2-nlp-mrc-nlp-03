
<p align="center">
  <a>
    <img alt="boostcampAItech" title="boostcampAItech" src="static/boostcamp.png">
  </a>
</p>

<div align="center">
<h1> Open-Domain Question Answering </h1> 

![Static Badge](https://img.shields.io/badge/NAVER-green?style=flat-square&logo=naver)
![Static Badge](https://img.shields.io/badge/Boostcamp-AI%20Tech-blue?style=flat-square)
![Static Badge](https://img.shields.io/badge/LEVEL2-NLP-purple?style=flat-square)
![Static Badge](https://img.shields.io/badge/3%EC%A1%B0-%EB%8F%99%ED%96%89-blue?style=flat-square&color=%09%23AAF0D1)

</div>


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

<p align="center">
  <img src = "static/odqa.png">
</p>

기계독해(MRC, Machine Reading comprehension)란 인공지능이 사람처럼 텍스트를 이해하고, 그 내용에 대해 질문에 대답하는 기술입니다. 이에 대한 하위 업무인 Question Answering(QA)은 가장 기본적인 MRC 태스크로, 주어진 문장이나 문단에 대한 질문에 대답하는 작업을 말합니다.  
이중에서도 본 대회의 주제인 Open-Domain QA(ODQA)는 질문과 지문이 함께 주어지지 않고 사전에 구축된 Knowledge resource에서 질문에 답하는데 필요한 문서를 검색 및 참고하여 질문에 답변하는 보다 어려운 문제이지만, 모델이 참조하는 지식 원천을 직접 관리할 수 있고 보다 일관된 품질의 답변을 얻을 수 있다는 점에서 활발히 연구되고 있습니다.


## Features

* EDA
* Data preprocessing
* Fine-tuning (KorQuAD v1.0)
* BM25
* DPR
* ColBERT



## Installation

```bash
cd # HOME으로 이동
git clone git@github.com:boostcampaitech6/level2-nlp-mrc-nlp-03
cd level2-nlp-mrc-nlp-03
```

#### Dependencies 설치

```bash
# conda 사용
conda env create -f environment.yaml
conda activate mrc

# mamba 사용
mamba env create -f environment.yaml
mamba activate mrc
```

#### Commit 환경 설정

```bash
make setup # commit message 설정 및 pre-commit 관련 패키지를 설치
```

## How To Run

#### 실행위치

```bash
$ pwd
/data/ephemeral/home/level2-nlp-mrc-nlp-03
```

#### Train

[script/train.sh](script/train.sh)에서 설정을 간편하게 변경할 수 있습니다.

```bash
make train # sh script/train.sh
```

#### Inference

[script/train.sh](script/inference.sh)에서 설정을 간편하게 변경할 수 있습니다.

```bash
make predict # sh script/inference.sh
```

#### Pipeline

train -> evaluation -> inference

```bash
make run # sh script/train.sh && sh script/inference.sh
```

## How To Pre-commit

```bash
git add <commit할 file> # 커밋하고 싶은 파일을 stage에 추가
git commit # pre-commit 실행
```

프리커밋은 커밋에 앞서 주로 코드 포맷, 문법 오류 또는 오탈자를 점검합니다. 커밋할 파일에 이에 해당되는 사항이 있다면 커밋 명령이 취소됩니다. pre-commit이 알아서 코드를 수정해주는 부분도 있고 혹은 직접 수정해줘야하는 경우도 있습니다. 커밋이 허가될 때까지 앞선 과정을 반복해주면 됩니다. 커밋 요청이 정상적으로 받아들여지면 commit message를 입력할 수 있는 윈도우가 나타납니다 (.gitmessage 참고).


## Contributors

<table align='center'>
  <tr>
    <td align="center">
      <img src="https://github.com/dustnehowl.png" alt="김연수" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/dustnehowl">
        <img src="https://img.shields.io/badge/%EA%B9%80%EC%97%B0%EC%88%98-grey?style=for-the-badge&logo=github" alt="badge 김연수"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/jingi-data.png" alt="김진기" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/jingi-data">
        <img src="https://img.shields.io/badge/%EA%B9%80%EC%A7%84%EA%B8%B0-grey?style=for-the-badge&logo=github" alt="badge 김진기"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/SeokSukyung.png" alt="석수경" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/SeokSukyung">
        <img src="https://img.shields.io/badge/%EC%84%9D%EC%88%98%EA%B2%BD-grey?style=for-the-badge&logo=github" alt="badge 석수경"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/Secludor.png" alt="오주영" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Secludor">
        <img src="https://img.shields.io/badge/%EC%98%A4%EC%A3%BC%EC%98%81-grey?style=for-the-badge&logo=github" alt="badge 오주영"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/gyunini.png" alt="이균" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/gyunini">
        <img src="https://img.shields.io/badge/%EC%9D%B4%EA%B7%A0-grey?style=for-the-badge&logo=github" alt="badge 이균"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/Yeonseolee.png" alt="이서연" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Yeonseolee">
        <img src="https://img.shields.io/badge/%EC%9D%B4%EC%84%9C%EC%97%B0-grey?style=for-the-badge&logo=github" alt="badge 이서연"/>
      </a> 
    </td>
  </tr>
</table>


## Reference
[1] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.  
[2] sentence-aware contrastive learning for Open-Domain Passage Retrieval  
[3] Wu, B., Zhang, Z., Wang, J., & Zhao, H. (2021). Sentence-aware contrastive learning for open-domain passage retrieval. arXiv preprint arXiv:2110.07524.  
[4] Khattab, O., & Zaharia, M. (2020, July). Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 39-48).  
[5] Khattab, O., Potts, C., & Zaharia, M. (2021). Relevance-guided supervision for openqa with colbert. Transactions of the association for computational linguistics, 9, 929-944.  
[6] Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2021). Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488.  
[7] stanford-futuredata. (n.d.). ColBERT: State-of-the-art neural search. GitHub. Retrieved from https://github.com/stanford-futuredata/ColBERT  
[8] bclavie. (2024). RAGatouille. GitHub. Retrieved from https://github.com/bclavie/ragatouille  
[9] deepset GmbH. (2024). Haystack. Retrieved from https://haystack.deepset.ai/   
[10] Weng, Lilian. (Oct 2020). How to build an open-domain question answering system? Lil’Log. https://lilianweng.github.io/posts/2020-10-29-odqa/.  
[11] Park, S., Moon, J., Kim, S., Cho, W. I., Han, J., Park, J., ... & Cho, K. (2021). Klue: Korean language understanding evaluation. arXiv preprint arXiv:2105.09680.   
[12] 임승영, 김명지, & 이주열. (2018). KorQuAD: 기계독해를 위한 한국어 질의응답 데이터셋. 한국정보과학회 학술발표논문집, 539-541.  
[13] 김영민, 임승영, 이현정, 박소윤, & 김명지. (2020). KorQuAD 2.0: 웹문서 기계독해를 위한 한국어 질의응답 데이터셋. 정보과학회논문지, 47(6), 577-586.  
