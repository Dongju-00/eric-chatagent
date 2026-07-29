# Changelog

## [1.1.0] - 2026-07-11
### Added
- LangGraph 기반 라우팅 Agent (smalltalk / stock_rag / fallback)
- 도메인별 파인튜닝 모델 분리 (KoreanGPT_smalltalk, KoreanGPT_news)

### Changed
- 사전학습 데이터 확대 (34M → 430M 토큰, 위키 + 나무위키)
- 모델 규모 상향 (23M → 42M 파라미터, block_size 256 → 384)

### Fixed
- 라우터 출력의 백틱으로 인한 분기 오류
- baseline.py import 시 평가 파이프라인이 실행되던 문제

## [1.0.0] - 2026-07-07
- 초기 RAG 파이프라인 (LangChain 기반)


git clone을 진행하니까 git에 모델, 모델 가중치, 토크나이저가 없어 오류가 나왔습니다

그래서 로컬에서 ec2 서버로 바로 옮기는

scp -i xxx.pem -r src/model/checkpoints/*.pt \
    ubuntu@ec2-54-180-145-23.ap-northeast-2.compute.amazonaws.com:~/eric-chatbot/src/model/checkpoints

를 진행했습니다 같은 방식으로 .model, .vocab 파일도 넣었습니다.

main.py에 host 주소를 127.0.0.1로 로컬로만 잡혀 있어 0.0.0.0으로 변경했습니다