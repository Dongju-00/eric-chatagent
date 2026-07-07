# KoreanGPT 한국어 챗봇

직접 구현한 Decoder-only Transformer 기반 한국어 챗봇 프로젝트입니다.  
SentencePiece 토크나이저를 학습한 뒤, 한국어 말뭉치로 사전학습을 진행하고, Q&A 대화 데이터로 추가 파인튜닝합니다.  
학습된 모델은 FastAPI로 감싸 `/chat` API를 통해 사용할 수 있습니다.

## 주요 기능

- SentencePiece 기반 한국어 BPE 토크나이저 학습
- Decoder-only GPT 구조 직접 구현
- Stage 1: 한국어 말뭉치 기반 사전학습
- Stage 2: 질문-답변 데이터 기반 파인튜닝
- FastAPI 기반 챗봇 API 제공


## 프로젝트 구조

```text
src/
  app/
    app.py                    # FastAPI
  model/
    main.py                   
    model.py                  # 하이퍼 파라미터 설정 및 tansform 모델 구성
    chat.py
    tokenizer.py              # 
    sp_tokenizer.py
    train_utils.py            #
    scripts/
      prepare_pretrain.py     # 사전학습 데이터 전처리 
      prepare_finetune.py     # 파인튜닝 데이터 전처리
    data/                     
    checkpoints/
      KoreanGPT.pt            # 한국어 사전학습 모델
      KoreanGPT_qa.pt         # 파인튜닝 된 모델 
requirements.txt
```

## 사용 기술

* Python
* PyTorch
* FastAPI
* Uvicorn
* SentencePiece
* Hugging Face Datasets
* Pandas

## 데이터 출처

| 용도       | 데이터셋                                                                                                             | 설명                    |
| -------- | ---------------------------------------------------------------------------------------------------------------- |-----------------------|
| 사전학습     | [HAERAE-HUB/KOREAN-WEBTEXT](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT)                           | 한국어 웹 텍스트 말뭉치         |
| 사전학습     | [heegyu/namuwiki-extracted](https://huggingface.co/datasets/heegyu/namuwiki-extracted)                           | 나무위키 기반 한국어 텍스트       |
| Q&A 파인튜닝 | [NLPBada/korean-persona-chat-dataset-v2](https://huggingface.co/datasets/NLPBada/korean-persona-chat-dataset-v2) | 한국어 페르소나 대화 데이터       |
| Q&A 파인튜닝 | `smalltalk_all.csv`                                                                                              | 생성한 소규모 일상 대화 Q&A 데이터 |

데이터 원본의 라이선스와 사용 조건은 각 데이터셋 페이지를 기준으로 따릅니다.

## 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사전학습 데이터 준비

```bash
cd src/model
python scripts/prepare_pretrain.py
```

실행 후 아래 파일이 생성됩니다.

```text
data/processed/pretrain.txt
```

## Stage 1 사전학습

```bash
python main.py train
```

실행 후 아래 모델 파일이 생성됩니다.

```text
checkpoints/KoreanGPT.pt
```

## Q&A 파인튜닝 데이터 준비

```bash
python scripts/prepare_finetune.py
```

실행 후 아래 파일이 생성됩니다.

```text
data/processed/qa_train.csv
data/processed/qa_val.csv
```

## Stage 2 Q&A 파인튜닝

```bash
python main.py train_qa
```

실행 후 아래 모델 파일이 생성됩니다.

```text
checkpoints/KoreanGPT_qa.pt
```

## CLI 테스트

```bash
python main.py chat_qa
```

예시:

```text
질문: 안녕
답변: 안녕! 반가워.
```

## FastAPI 실행

`app` 폴더에서 실행합니다.

```bash
uvicorn app:app --reload --port 8000
```

상태 확인:

```bash
curl http://127.0.0.1:8000/chat
```

채팅 요청 예시:

```json
{
  "message": "안녕"
}
```

응답 예시:

```json
{
  "reply": "안녕! 반가워."
}
```

## 현재 한계

현재 모델은 요청 하나에 대해 하나의 답변을 생성하는 단일턴 구조입니다.
이전 대화 내용을 별도로 저장하거나 검색하지 않기 때문에 멀티턴 기억은 지원하지 않습니다.

멀티턴 대화를 지원하려면 `session_id` 기반 대화 기록 저장 또는 VectorDB 기반 메모리 검색 구조를 추가해야 합니다.

## 향후 개선 계획

* VectorDB 기반 대화 메모리 저장 및 검색
* `session_id` 기반 멀티턴 대화 지원
* 답변 품질 개선을 위한 추가 파인튜닝

```gitignore
.venv/
__pycache__/
*.pyc

# large training data
src/model/data/raw_hf/
src/model/data/processed/
src/model/data/raw_aihub/

# generated tokenizer
src/model/data/sp_korean.model
src/model/data/sp_korean.vocab

# model checkpoints
src/model/checkpoints/
```

## 실행 순서 요약

```bash
cd src/model

python scripts/prepare_pretrain.py
python main.py train

python scripts/prepare_finetune.py
python main.py train_qa

cd ../app
uvicorn app:app --reload --port 8000
```

