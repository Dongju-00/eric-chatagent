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