# 측정

변경 전후를 같은 기준으로 비교하기 위한 디렉터리다. 측정하지 않은 개선은
이 프로젝트의 산출물로 보지 않는다.

하네스는 두 개다.

| 스크립트 | 보는 것 |
|---|---|
| `measure_retrieval.py` | 뉴스 수집·검색 품질과 속도 |
| `measure_startup.py` | 서빙 시작 비용 (가중치 적재 횟수, Chroma 클라이언트 수, 시간, RSS) |

## 시작 비용 측정

```bash
LANGSMITH_TRACING=false PYTHONPATH=src uv run python measurements/measure_startup.py
```

결과는 `results/startup_after.json` 에 남는다. 변경 전 기준선은 스펙에 기록된
`import app.app` 한 번당 `torch.load` 4회 / `chromadb.PersistentClient` 2개다.

RSS는 참고용으로만 본다. KoreanGPT 1벌의 fp32 파라미터는 159.5MB지만, macOS의
메모리 압축과 할당자 재사용 때문에 측정된 RSS가 적재 벌수에 비례하지 않는다.
전후 비교가 가능한 수치는 적재 횟수다.

## 검색 측정 실행

```bash
# 변경 전 기준선
LANGSMITH_TRACING=false PYTHONPATH=src uv run python measurements/measure_retrieval.py --label before

# 변경 후
LANGSMITH_TRACING=false PYTHONPATH=src uv run python measurements/measure_retrieval.py --label after

# 비교 표 출력
uv run python measurements/measure_retrieval.py --compare before after
```

결과는 `results/<label>.json` 에 원시 데이터로 남는다. 같은 label 로 다시
실행하면 덮어쓴다.

`LANGSMITH_TRACING=false` 는 유효하지 않은 LangSmith 키 때문에 요청마다
403 로그가 쏟아지는 것을 막기 위한 것이다.

## 측정 항목

| 항목 | 뜻 |
|---|---|
| `t_collect` | 수집 시간 — 네이버 API 호출 + 본문 추출 + 청킹 + 색인 |
| `t_retrieve` | 검색 시간 — 질문 임베딩 + Chroma 질의 |
| `company_resolved_rate` | 질문에서 종목이 식별된 비율. 종목 사전 커버리지를 본다 |
| `company_hit_rate` | 검색된 청크가 대상 종목명(별칭 포함)을 포함하는 비율 |
| `age_days` | 검색된 청크 기사의 경과 일수. 최신성을 본다 |
| `empty_context_rate` | 컨텍스트가 0건인 질문 비율 |
| `distance_median` | Chroma 거리. 임베더 변별력을 본다 |

### 평가 질문 구성

`measure_retrieval.py` 의 `QUESTIONS` 10개는 의도적으로 두 종류의 어려운
케이스를 포함한다.

- `카카오 실적 어때?` — 종목 사전에 없는 기업. 사전 커버리지 구멍을 드러낸다.
- `삼전 주가 전망` — 별칭으로만 부르는 질문.

질문을 바꾸면 전후 비교가 깨진다. 비교 대상 측정은 같은 질문 집합으로 돌려야 한다.

## 주의

- 네이버 뉴스 API를 실제로 호출하고 기사 본문을 가져온다. 뉴스가 바뀌면
  같은 코드로도 수치가 흔들린다. 전후 측정은 되도록 가까운 시점에 돌린다.
- 매 실행이 `storage/chroma` 에 질문과 청크를 추가한다. 측정을 반복하면
  `user_questions` 컬렉션이 커진다.
- `company_hit_rate` 는 청크 텍스트에 종목명이 들어있는지만 본다. 기사가
  실제로 그 종목을 다루는지에 대한 근사치이며, 사람 판단과 완전히 같지는 않다.
  청크 앞머리에 `제목:` 이 붙으므로 제목 일치는 반영된다.

## 멀티턴 fine-tuning 데이터 및 모델 전후 기록

```bash
uv run --locked python measurements/measure_finetune_data.py \
  --output measurements/results/finetune_data_audit_<새라벨>.json
```

위 명령은 데이터 구조만 실측한다. 모델의 학습 전후 성능은 수정된 Colab 노트북이 동일 평가셋으로 측정해 Drive의 각 models/실행폴더에 `before_metrics.json`, `after_metrics.json`, `comparison.json/.csv`, `manual_review.csv`로 기록한다. 자세한 지표 정의와 한계는 [측정·멀티턴 확인 보고서](../docs/reports/2026-10-24-finetuning-measurement-and-multiturn.md)를 참고한다. GPU 학습 전에는 실제 모델 향상 수치를 보고하지 않는다.

실제 첫 Colab 실행의 전후 수치는 `results/colab_20261024_first_run/`에 보존했고, [결과 해석](../docs/reports/2026-10-24-finetuning-first-results.md)에 loss 개선과 멀티턴 실패를 함께 기록했다.
