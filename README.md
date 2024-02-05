## Installation

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

## How To Pre-commit

```bash
git add <commit할 file> # 커밋하고 싶은 파일을 stage에 추가
git commit # pre-commit 실행
```

프리커밋은 커밋에 앞서 주로 코드 포맷, 문법 오류 또는 오탈자를 점검합니다. 커밋할 파일에 이에 해당되는 사항이 있다면 커밋 명령이 취소됩니다. pre-commit이 알아서 코드를 수정해주는 부분도 있고 혹은 직접 수정해줘야하는 경우도 있습니다. 커밋이 허가될 때까지 앞선 과정을 반복해주면 됩니다. 커밋 요청이 정상적으로 받아들여지면 commit message를 입력할 수 있는 윈도우가 나타납니다 (.gitmessage 참고).
