# Changelog

이 프로젝트의 주요 변경 사항을 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따르며, 버전은 [유의적 버전](https://semver.org/lang/ko/)을 사용합니다.

<br>

## [1.3.0] - 2026-07-30

### Added
- GitHub Actions CI/CD 파이프라인 (`test` → `build-and-push-image` → `deploy`)
- Hugging Face Hub 기반 모델 배포 ([Dongju-00/eric-chatagent](https://huggingface.co/Dongju-00/eric-chatagent))
  - 이미지 빌드 시 가중치·토크나이저를 자동 다운로드
- EC2 Docker 자동 설치 및 Docker Hub 로그인 단계 (신규 인스턴스 대응)
- 서버 초기 세팅 체크리스트 문서화

### Changed
- PyTorch를 CPU 전용 빌드로 고정 (`pyproject.toml`의 `tool.uv.sources`)
  - 이미지 크기 8.6GB → 약 1.5GB, 빌드·배포 시간 대폭 단축
- 모델 파일 관리 방식을 수동 전송(`scp`)에서 Hub 다운로드로 전환
- 배포용 `docker-compose.yml`에서 `build: .` 제거 (배포 서버는 pull만 수행)
- Dockerfile `WORKDIR`를 `/`에서 `/app`으로 변경, `ENV PATH` 경로 정합성 수정

### Fixed
- CI 빌드 이미지에 모델 파일이 누락되어 컨테이너가 기동하지 못하던 문제
- Docker 이미지 빌드 태그와 push 태그 불일치로 인한 배포 실패
- `t3.small` 환경에서 모델 3종 동시 로드로 OOM(exit 137)이 발생하던 문제

### Security
- 노출된 SSH 키 및 Docker Hub 토큰 폐기 후 재발급
- `.env`를 `.dockerignore` / `.gitignore`에 포함하여 이미지·저장소 유입 차단
- 배포용 비밀값(GitHub Secrets)과 실행용 비밀값(서버 `.env`) 분리

<br>

## [1.2.0] - 2026-07-28

### Added
- Docker 컨테이너화 (`Dockerfile`, `docker-compose.yml`)
- AWS EC2 배포 환경 구성
- 웹 채팅 UI (`src/app/static/index.html`, `app.js`)
  - 세션 유지, 입력 중 표시, Enter 전송 지원
- FastAPI 정적 파일 서빙 (`StaticFiles` 마운트)
- `POST /agent/chat` 응답에 `route`, `trace`, `contexts` 필드 추가 (디버깅용)

### Changed
- 서버 바인딩 주소를 `127.0.0.1`에서 `0.0.0.0`으로 변경 (외부 접속 허용)
- 정적 파일 경로를 실행 위치가 아닌 파일 기준 절대 경로로 계산

### Fixed
- 상대 경로로 인해 실행 위치에 따라 정적 파일을 찾지 못하던 문제

<br>

## [1.1.0] - 2026-07-11

### Added
- LangGraph 기반 라우팅 Agent (smalltalk / stock_rag / fallback)
- 도메인별 파인튜닝 모델 분리 (`KoreanGPT_smalltalk`, `KoreanGPT_news`)

### Changed
- 사전학습 데이터 확대 (34M → 430M 토큰, 위키 + 나무위키)
- 모델 규모 상향 (23M → 42M 파라미터, `block_size` 256 → 384)
- 어텐션 연산을 PyTorch 내장 최적화 어텐션으로 교체 (causal mask 자동 처리)

### Fixed
- 라우터 출력의 백틱으로 인한 분기 오류
- `baseline.py` import 시 평가 파이프라인이 실행되던 문제

<br>

## [1.0.0] - 2026-07-07

### Added
- 초기 RAG 파이프라인 (LangChain 기반)
- KoreanGPT 사전학습 및 SentencePiece 토크나이저
- 네이버 뉴스 검색 연동, ChromaDB 벡터 스토어