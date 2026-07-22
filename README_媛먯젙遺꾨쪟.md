# 감정분류 모델 (ML 파트) — 완료본

프로젝트: **종목 뉴스 심리 분석 + 트레이딩 시나리오 AI**
이 폴더는 그중 "직접 학습한 ML 모델" 부분이야. 학습·검증까지 다 끝냈고, 네 컴에선 실행/통합만 하면 돼.

## 파일

| 파일 | 설명 |
|---|---|
| `train_sentiment.py` | 학습 스크립트 (모델 3개 비교 후 최적 저장) |
| `sentiment.py` | 추론 모듈 — 트레이딩 봇에 import 해서 쓸 부분 |
| `sentiment_model.pkl` | 학습된 모델 (내 환경 결과, 참고/백업용) |
| `metrics.txt` | 발표용 정확도·리포트 |
| `data/Sentences_*.txt` | Financial PhraseBank 원본 (오프라인 학습용) |

## 결과 (발표용 숫자)

- 데이터: Financial PhraseBank 50%합의 세트, 영어 금융뉴스 4,846문장 (중립 2879 / 긍정 1363 / 부정 604)
- 방식: **TF-IDF(단어 1~2gram + 문자 3~5gram) → 로지스틱회귀 / 선형SVM / 나이브베이즈 비교**
- **최종 선택: LinearSVC — 정확도 77.9%, macro-F1 0.729** (test 20%)
- 한계(정직하게): 중립 데이터가 많아 **중립 편향** 존재 → 소수 클래스(부정) recall이 낮음. 경량·고속이 장점, 정밀도는 BERT 계열이 우위. → "왜 딥러닝 안 썼나" 질문 대비 논점으로 활용.

## 네 컴에서 할 일

```bash
# 1) 패키지 (venv 안에서)
pip install scikit-learn pandas joblib

# 2) 모델 학습 (약 30초, 네 환경 sklearn 버전에 맞는 pkl 새로 생성 — 권장)
python train_sentiment.py

# 3) 추론 테스트
python sentiment.py
```

> ⚠️ 동봉한 `sentiment_model.pkl`은 내 환경(sklearn 1.8.0)에서 만든 거라, 네 sklearn 버전이 다르면 로드 경고가 날 수 있어. **train_sentiment.py를 한 번 돌려서 네 환경 pkl을 새로 뽑는 걸 추천.** 데이터 원본이 `data/`에 있어서 인터넷 없이도 학습돼.

## 트레이딩 봇과 통합 (다음 단계)

`sentiment.py`의 `analyze_news()`가 종합 심리 점수(0~100)를 주니까, 이걸 기존 봇 프롬프트에 주입하면 돼:

```python
from sentiment import analyze_news

news = [...]                      # NewsAPI로 받은 종목 뉴스 제목/요약 리스트
s = analyze_news(news)
# s["sentiment_score"] -> 예: 58.6,  s["verdict"] -> "중립/혼조",  s["counts"], s["items"]

prompt = f"""
종목 뉴스 심리 분석 결과: 심리점수 {s['sentiment_score']}/100 ({s['verdict']}),
분포 {s['counts']}. 시드머니 {seed}, 목표수익 {target}.
위 뉴스 심리를 반영해 진입/청산 관점의 트레이딩 시나리오를 제시해줘.
"""
# 이 prompt를 기존 GPT-5 mini 호출부에 넣으면 끝
```

## 남은 로드맵

1. ✅ 감정분류 모델 학습 — **완료**
2. ⬜ NewsAPI로 종목 뉴스 수집 (newsapi.org 무료키, 미국주식)
3. ⬜ `analyze_news()`를 FastAPI에 연결
4. ⬜ 심리 점수를 트레이딩 봇 프롬프트에 주입
5. ⬜ Streamlit UI (심리 그래프 + 시나리오)
6. ⬜ 발표 리허설 (7.24)
