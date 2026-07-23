# 뉴스 심리 기반 트레이딩 시나리오 AI

종목 티커를 넣으면 →  **뉴스 수집 → 감정분류 ML로 심리점수 → 준실시간 시세 → GPT-5.4가 단기 매매 시나리오**를 생성하는 웹앱.

```
티커(TSLA)
  │
  ├─ 뉴스 수집        news_fetch.py   (NewsAPI, 금융매체 우선)
  ├─ 감정분류 ML      sentiment.py    (FinBERT / 자체 학습 모델, 심리점수 0~100)
  ├─ 준실시간 시세     price_fetch.py  (yfinance, 약 15분 지연, OHLCV + 환율)
  └─ 시나리오 생성     main.py         (엘리스 프록시로 GPT-5.4 호출)
        │
        └─ 화면        app.py          (Streamlit: 캔들차트 + 심리 + 시나리오)
```

---

## 실행 방법

```bash
# 1) 패키지 설치 (venv 안에서)
pip install -r requirements.txt

# 2) .env 준비  (.env.example 복사 후 키 입력)
#    NEWSAPI_KEY, ELICE_API_KEY, ELICE_BASE_URL, ELICE_MODEL, SENTIMENT_ENGINE

# 3) 백엔드 실행 (터미널 1)
uvicorn main:app --reload

# 4) 프런트 실행 (터미널 2)
streamlit run app.py
#    → http://localhost:8501
```

> FinBERT는 첫 실행 때 모델(~440MB)을 자동 다운로드 후 캐시. 이후엔 오프라인도 동작.

---

## 파일

| 파일 | 설명 |
|---|---|
| `app.py` | **Streamlit UI** — 캔들차트/거래량/뉴스심리 오버레이, 심리점수, 시나리오 |
| `main.py` | **FastAPI 백엔드** — 뉴스+심리+시세를 묶어 GPT-5.4 시나리오 생성 |
| `sentiment.py` | 감정분류 추론 (FinBERT 기본 + 자체 모델 폴백/비교) |
| `news_fetch.py` | NewsAPI 뉴스 수집 (티커→회사명, 금융매체 필터) |
| `price_fetch.py` | yfinance 가격/시세/환율 (OHLCV, 1일·1달·3달·1년) |
| `train_sentiment.py` | 자체 감정분류 모델 학습 스크립트 |
| `sentiment_model.pkl` | 학습된 자체 모델 (TF-IDF + LinearSVC) |
| `run_pipeline.py` | CLI 데모 (`--mock` 오프라인 테스트 지원) |
| `metrics.txt` | 자체 모델 정확도·리포트 |
| `data/Sentences_*.txt` | Financial PhraseBank 원본 (오프라인 학습용) |

---

## 주요 기능

### 1. 가격 차트 (app.py)
- **캔들스틱** (상승 초록 / 하락 빨강) + **거래량 막대** + **전일 종가 기준선**
- 기간 탭: **1일 / 1달 / 3달 / 1년**, 전일대비 등락 표시
- **뉴스 심리 오버레이**: 각 뉴스를 발행일·가격 위에 감정별 마커(▲긍정 ▼부정 ●중립)로 표시
- **원화 병기**: 달러 가격에 환율(USD/KRW) 적용한 원화 표시
- 조작: 드래그=이동 · 스크롤=확대/축소 · 더블클릭=원상복구

### 2. 감정분류 — 두 엔진 (sentiment.py)
`SENTIMENT_ENGINE` 환경변수로 전환 (`finbert` 기본 | `sklearn`).

| 엔진 | 설명 | 정확도 |
|---|---|---|
| **FinBERT** | 금융 특화 사전학습 BERT (ProsusAI/finbert) | 약 88~90% |
| **자체 학습 모델** | TF-IDF(단어+문자 n-gram) + LinearSVC — **직접 학습** | 77.9% |

- `analyze_news(texts)` → 종합 심리점수(0~100) + 분포 + 개별결과 + 엔진정보
- 심리점수 = 각 뉴스의 `P(긍정) - P(부정)` 평균을 0~100으로 변환 (50=중립)

### 3. 준실시간 시세 → 시나리오 (price_fetch.py, main.py)
- yfinance로 현재가·당일 고저·전일종가 조회 (**약 15분 지연**, 무료)
- 이 실제 가격을 프롬프트에 주입 → 시나리오가 구체적 진입가·목표가·손절가를 **실제 숫자**로 제시
- 완전 실시간은 유료 피드(Polygon/Alpaca 등) 필요

---

## 자체 학습 모델 상세 (발표용)

- **데이터**: Financial PhraseBank (Malo et al., 2014), 영어 금융뉴스. `Sentences_50Agree.txt` 4,846문장 (중립 2879 / 긍정 1363 / 부정 604)
- **방식**: TF-IDF(단어 1~2gram + 문자 3~5gram) → 로지스틱회귀 / 선형SVM / 나이브베이즈 비교 → **LinearSVC 선택** (정확도 77.9%, macro-F1 0.729)

### 데이터 동의율별 정확도 (같은 파이프라인, DATA_FILE만 변경)

| 데이터셋 | 문장수 | 정확도 | macro-F1 |
|---|---|---|---|
| 50Agree (기본) | 4,846 | 77.9% | 0.729 |
| 66Agree | 4,217 | 81.9% | 0.772 |
| 75Agree | 3,453 | 86.8% | 0.817 |
| AllAgree (만장일치) | 2,264 | 92.1% | 0.888 |

> 50Agree는 주석자 절반만 동의한 **가장 노이즈 많은** 세트라 상한이 낮음. 동의율이 높을수록(라벨이 깨끗) 정확도가 오름. → "왜 77%냐" 질문에 데이터 품질 관점으로 답할 수 있음.
>
> **한계(정직하게)**: 중립 데이터가 많아 중립 편향 존재, 소수 클래스(부정) recall이 낮음. 경량·고속이 장점. 정밀도는 BERT 계열이 우위 → 그래서 앱 기본 엔진은 FinBERT, 자체 모델은 비교·폴백용.

### 자체 모델 재학습
```bash
python train_sentiment.py     # 약 30초, 네 환경 sklearn 버전에 맞는 pkl 새로 생성
python sentiment.py           # 추론 테스트
```

---

## API 엔드포인트 (main.py)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 헬스체크 (사용 모델 반환) |
| POST | `/analyze` | 심리분석만 (미리보기용) |
| GET | `/price/{ticker}?period=1달` | 가격 차트 데이터 (OHLCV + 전일대비 + 환율) |
| GET | `/quote/{ticker}` | 준실시간 시세 (현재가·당일고저·전일종가·환율) |
| POST | `/generate` | 뉴스심리 + 시세 + 조건 → GPT-5.4 시나리오 |

---

## 환경변수 (.env)

```ini
NEWSAPI_KEY=...            # newsapi.org 무료키
ELICE_API_KEY=...          # 엘리스 AI 클라우드 (OpenAI 호환 프록시)
ELICE_BASE_URL=...
ELICE_MODEL=openai/gpt-5.4
SENTIMENT_ENGINE=finbert   # finbert(기본) | sklearn(자체 모델)
```

> `.env` 는 `.gitignore` 에 등록되어 깃허브에 올라가지 않음.

---

## 로드맵

1. ✅ 감정분류 모델 학습 (자체 + FinBERT)
2. ✅ NewsAPI 뉴스 수집
3. ✅ FastAPI 백엔드 (`analyze_news` 연결)
4. ✅ 심리 점수 + 준실시간 시세를 트레이딩 봇 프롬프트에 주입
5. ✅ Streamlit UI (캔들차트 + 심리 오버레이 + 시나리오)
6. ⬜ 발표 리허설 (7.24)

> ⚠️ 교육·시뮬레이션 목적의 예시이며 실제 투자 조언이 아님.
