# eric-chatagent

한국어 대화 모델을 직접 학습하고, 네이버 뉴스 검색 기반 RAG와 LangGraph 라우팅을 연결한 주식 정보 챗봇입니다.
사전학습 모델을 가져다 쓰는 대신 **Decoder-only Transformer(KoreanGPT)를 직접 설계·학습**하고, 여기에 검색 증강 생성과 서비스 배포까지 붙였습니다.

<br>

## 목차

- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
- [API](#api)
- [배포 파이프라인](#배포-파이프라인)
- [서버 초기 세팅](#서버-초기-세팅)
- [모델](#모델)
- [알려진 한계](#알려진-한계)

<br>

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| **질문 라우팅** | LangGraph 라우터가 질문을 분류해 세 경로 중 하나로 분기 |
| **스몰톡** | 일상 대화 전용으로 파인튜닝한 KoreanGPT가 응답 |
| **주식 뉴스 RAG** | 네이버 뉴스 수집 → 벡터 인덱싱 → 유사도 검색 → 뉴스 전용 모델이 응답 |
| **폴백** | 처리 범위를 벗어난 질문은 정형 응답으로 안내 |
| **세션 유지** | `thread_id` 기반으로 대화 상태를 이어서 관리 |
| **웹 채팅 UI** | 카카오톡 스타일 채팅 인터페이스 제공 |

<br>

## 아키텍처

```
                        ┌─────────────────────────┐
   브라우저  ──────────▶  │  FastAPI (:8000)        │
   (채팅 UI)             │  · 정적 파일 서빙          │
                        │  · POST /agent/chat     │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │      LangGraph          │
                        │      router_node        │
                        └───┬───────┬─────────┬───┘
                            │       │         │
                  smalltalk │       │ stock_rag         fallback
                            │       │         │
              ┌─────────────▼─┐  ┌──▼──────────────┐  ┌▼──────────┐
              │ KoreanGPT     │  │ search_news     │  │ 정형 응답   │
              │ _smalltalk    │  │   ↓ 네이버 API    │  └───────────┘
              └───────────────┘  │ retrieve_news   │
                                 │   ↓ ChromaDB    │
                                 │ generate        │
                                 │   ↓ KoreanGPT   │
                                 │     _news       │
                                 └─────────────────┘
```

**RAG 흐름**: 주식 질문이 들어오면 네이버 뉴스를 수집해 `KoreanGPT.pt`(임베딩 전용)로 벡터화하고 ChromaDB에 저장합니다. 같은 임베더로 질문을 벡터화해 유사 뉴스를 검색한 뒤, 검색 결과를 컨텍스트로 넣어 `KoreanGPT_news.pt`가 답변을 생성합니다.

<br>

## 기술 스택

**모델**
`PyTorch` · `SentencePiece` · Decoder-only Transformer (자체 구현)

**RAG / 오케스트레이션**
`LangChain` · `LangGraph` · `ChromaDB` · 네이버 뉴스 검색 API

**서빙**
`FastAPI` · `Uvicorn` · Vanilla JS 채팅 UI

**인프라**
`Docker` · `Docker Compose` · `GitHub Actions` · `AWS EC2` · `Docker Hub` · `Hugging Face Hub`

**패키지 관리**
`uv`

<br>

## 프로젝트 구조

```
eric-chatagent/
├── main.py                      # 진입점 (uvicorn 실행)
├── pyproject.toml               # 의존성 정의 (torch는 CPU 빌드 고정)
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD 파이프라인
└── src/
    ├── app/
    │   ├── app.py               # FastAPI 앱, 엔드포인트
    │   └── static/              # 채팅 UI (index.html, app.js)
    └── model/
        ├── train/               # 모델 정의, SentencePiece 토크나이저
        ├── embed/               # KoreanGPT 기반 임베더
        ├── store/               # ChromaDB 벡터 스토어
        ├── rag/                 # 검색, LLM 래퍼, 프롬프트
        ├── graph/               # LangGraph 그래프 정의
        ├── data/                # 토크나이저 파일 (HF에서 주입)
        └── checkpoints/         # 모델 가중치 (HF에서 주입)
```

> `data/`, `checkpoints/`의 실제 파일은 Git에 포함되지 않습니다. 이미지 빌드 시 Hugging Face Hub에서 내려받습니다. ([모델](#모델) 참고)

<br>

## 시작하기

### 요구 사항

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- 네이버 검색 API 키

### 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/Dongju-00/eric-chatagent.git
cd eric-chatagent

# 2. 의존성 설치
uv sync

# 3. 모델 파일 다운로드 (Hugging Face)
uv run python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Dongju-00/eric-chatagent', local_dir='.')"

# 4. 환경 변수 설정
cp .env.example .env      # 네이버 API 키 등을 입력

# 5. 서버 실행
uv run python main.py
```

- 채팅 UI: http://localhost:8000
- API 문서: http://localhost:8000/docs

### Docker로 실행

```bash
docker compose up -d
```

<br>

## API

### `POST /agent/chat`

질문을 라우팅해 답변을 반환합니다.

**Request**

```json
{
  "question": "삼성전자 주식 뉴스 알려줘",
  "thread_id": "session-f703c100"
}
```

`thread_id`를 생략하면 새 세션이 발급됩니다. 이후 요청에 같은 값을 넘기면 대화 상태가 유지됩니다.

**Response**

```json
{
  "question": "삼성전자 주식 뉴스 알려줘",
  "answer": "제공된 참고 뉴스에 따르면 ...",
  "route": "stock_rag",
  "trace": ["router_node", "search_news_node", "retrieve_news_node", "generate_node"],
  "contexts": ["제목: ... \n내용: ... \n링크: ..."],
  "question_id": "q_20260729_021237_65e9df64",
  "thread_id": "session-f703c100"
}
```

| 필드 | 설명 |
| --- | --- |
| `route` | 라우팅 결과 (`smalltalk` / `stock_rag` / `fallback`) |
| `trace` | 그래프가 거친 노드 목록 |
| `contexts` | RAG가 참고한 뉴스 원문 |

### `GET /health`

서버와 그래프 로딩 상태를 반환합니다.

<br>

## 배포 파이프라인

`main` 브랜치에 push하면 전체 배포가 자동으로 진행됩니다.

```
git push origin main
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ GitHub Actions                                          │
│                                                         │
│  test                  uv sync --locked 검증             │
│    │ needs                                              │
│  build-and-push        Docker 이미지 빌드                  │
│    │                   (HF에서 모델 주입) → Docker Hub     │
│    │ needs                                              │
│  deploy                EC2 SSH 접속                      │
│                        → docker login                   │
│                        → compose pull                   │
│                        → compose up -d                  │
└─────────────────────────────────────────────────────────┘
      │
      ▼
   EC2에 새 버전 반영
```

### 필요한 GitHub Secrets

| 이름 | 설명 |
| --- | --- |
| `DOCKER_USERNAME` | Docker Hub 사용자명 |
| `DOCKER_TOKEN` | Docker Hub Access Token |
| `SERVER_HOST` | EC2 퍼블릭 주소 (Elastic IP 권장) |
| `SERVER_USER` | EC2 사용자명 (`ubuntu`) |
| `SSH_PRIVATE_KEY` | EC2 접속용 `.pem` 파일 전문 |

<br>

## 서버 초기 세팅

새 EC2 인스턴스에 배포할 때 필요한 작업입니다. **코드와 모델은 이미지에 포함되므로 서버에는 아래 항목만 준비하면 됩니다.**

```bash
# 1. 배포 디렉터리 생성 (EC2)
mkdir -p ~/your directory

# 2. 실행에 필요한 파일 전송 (로컬)
scp -i <key>.pem docker-compose.yml .env ubuntu@<host>:~/your directory/
```

**체크리스트**

- [ ] `~/your directory`에 `docker-compose.yml`, `.env` 존재
- [ ] compose 파일에 `build: .`이 **없을** 것 (배포 서버는 pull만 수행)
- [ ] 보안 그룹 인바운드: `22`(SSH), `8000`(웹)
- [ ] Elastic IP 연결 (재시작 시 주소 고정)
- [ ] 인스턴스 타입 `t3.medium` 이상 (모델 로딩에 메모리 필요)

> Docker 설치와 Docker Hub 로그인은 워크플로가 자동으로 처리합니다.

<br>

## 모델

학습한 모델 가중치와 토크나이저는 Hugging Face Hub에 공개되어 있습니다.

**[Dongju-00/eric-chatagent](https://huggingface.co/Dongju-00/eric-chatagent)**

| 파일 | 역할 |
| --- | --- |
| `KoreanGPT.pt` | 임베딩 생성 (RAG 벡터 검색용) |
| `KoreanGPT_news.pt` | 뉴스 기반 답변 생성 |
| `KoreanGPT_smalltalk.pt` | 스몰톡 답변 생성 |
| `sp_korean.model` / `.vocab` | SentencePiece 토크나이저 |

가중치는 Git에 커밋하지 않고 Hub에서 주입합니다. 덕분에 저장소는 코드만 유지하고, 로컬·CI·서버 어느 환경에서든 동일한 모델을 사용할 수 있습니다.

<br>

## 알려진 한계

- **환각(Hallucination)** — 소형 모델 특성상 검색된 뉴스에 없는 내용을 생성하는 경우가 있습니다. 컨텍스트 품질에 답변이 크게 좌우됩니다.
- **검색 품질** — 질문을 그대로 검색 쿼리로 사용하고 있어 종목과 무관한 기사가 섞일 수 있습니다. 종목명 추출과 관련도 임계값 도입이 필요합니다.
- **메모리 사용량** — 모델 3종을 시작 시 모두 로드하는 구조로, 2GB 환경에서 OOM이 발생합니다. 지연 로딩과 모델 서버 분리를 검토 중입니다.
- **문장 완성도** — 자체 학습한 소형 모델이라 조사·어미가 부자연스러운 출력이 나올 수 있습니다.

<br>

## 로드맵

- [ ] nginx 리버스 프록시 도입 (80 포트 서빙, 정적 파일 분리)
- [ ] 모델 추론을 별도 서버로 분리하여 장애 격리
- [ ] Gemini API 사용 부분을 공개 가중치 모델로 변경
- [ ] 검색 파이프라인 개선 (종목명 추출, `top_k` 확대, 관련도 임계값)
- [ ] 생성 파라미터 튜닝 (temperature 조정으로 환각 완화)
- [ ] HTTPS 적용
