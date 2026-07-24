# 뉴스 심리 기반 트레이딩 시나리오 AI

종목 티커와 투자 목표를 넣으면 →
**뉴스 수집 → 감정분류 ML로 심리점수 → 준실시간 시세·통계 → GPT-5.4가 목표 현실성 분석 + 매매 시나리오**를 생성하는 웹앱.

```
티커(TSLA) + 목표(기간·수익)
  │
  ├─ 뉴스 수집        news_fetch.py   (NewsAPI, 금융매체 우선 + 부족분 보충)
  ├─ 감정분류 ML      sentiment.py    (FinBERT / 자체 학습 모델, 심리점수 0~100)
  ├─ 준실시간 시세     price_fetch.py  (yfinance, 약 15분 지연, OHLCV·통계·환율)
  └─ 시나리오 생성     main.py         (엘리스 프록시로 GPT-5.4 호출)
        │
        └─ 화면        app.py          (Streamlit: 캔들차트 + 심리 + 목표분석 + 시나리오)
```

---

## 실행 방법 (로컬)

```bash
# 1) 패키지 설치 (venv 안에서)
pip install -r requirements.txt
# (선택) FinBERT까지 쓰려면 추가로:
pip install torch==2.4.1 transformers==4.48.3

# 2) .env 준비  (.env.example 복사 후 키 입력)
#    NEWSAPI_KEY, ELICE_API_KEY, ELICE_BASE_URL, ELICE_MODEL, SENTIMENT_ENGINE

# 3) 백엔드 실행 (터미널 1)
uvicorn main:app --reload

# 4) 프런트 실행 (터미널 2)
streamlit run app.py
#    → http://localhost:8501
```

> `requirements.txt`는 **배포용 최소 의존성**(torch/transformers 제외)이라 기본은 자체 학습 sklearn 모델로 동작.
> 로컬에서 FinBERT를 쓰려면 위처럼 torch/transformers를 추가 설치하고 `SENTIMENT_ENGINE=finbert`로 두면 됨.

---

## 배포 (Render, Docker)

무료 인스턴스(512MB RAM)에 맞춰 **배포판은 가벼운 자체 sklearn 모델만** 사용합니다.
(FinBERT/torch는 512MB에서 OOM이 나므로 배포에서 제외 — `Dockerfile`에 `SENTIMENT_ENGINE=sklearn` 고정)

1. render.com → **New → Web Service** → 이 GitHub 저장소 연결
2. **Language: Docker**, **Branch: main**, **Instance: Free**
3. **Environment Variables** 입력: `NEWSAPI_KEY`, `ELICE_API_KEY`, `ELICE_BASE_URL`, `ELICE_MODEL`
   (SENTIMENT_ENGINE은 Dockerfile에 박혀 있어 불필요)
4. **Deploy** → 로그에 `Your service is live` → `https://...onrender.com`

- `Dockerfile`: 빌드 시 `train_sentiment.py`로 배포 환경 sklearn 버전에 맞춰 모델을 새로 학습(실패 시 커밋된 pkl 사용)
- `start.sh`: FastAPI를 컨테이너 내부(127.0.0.1:8000)로, Streamlit만 `$PORT`로 외부 노출

> 무료 티어는 15분 미사용 시 잠들어 **첫 접속이 느립니다**(콜드 스타트).
> FinBERT까지 배포하려면 RAM이 넉넉한 유료 인스턴스나 **백엔드/프론트 분리**(백엔드=추론 서버, UI=Streamlit Cloud)가 필요.

---

## 파일

| 파일 | 설명 |
|---|---|
| `app.py` | **Streamlit UI** — 캔들차트/거래량, 목표·통계, 시나리오 |
| `main.py` | **FastAPI 백엔드** — 뉴스심리+시세+통계+투자목표 → GPT-5.4 |
| `sentiment.py` | 감정분류 추론 (FinBERT 기본 + 자체 모델 폴백/비교) |
| `news_fetch.py` | NewsAPI 뉴스 수집 (금융매체 우선 + 부족분 보충) |
| `price_fetch.py` | yfinance 가격/시세/통계/환율 (봉 단위: 일·주·월·분기·연) |
| `train_sentiment.py` | 자체 감정분류 모델 학습 스크립트 |
| `sentiment_model.pkl` | 학습된 자체 모델 (TF-IDF + LinearSVC) |
| `Dockerfile` · `start.sh` | 배포용 (Docker, 2-프로세스 구동) |
| `run_pipeline.py` | CLI 데모 (`--mock` 오프라인 테스트) |
| `metrics.txt` · `data/` | 자체 모델 리포트 · 학습 데이터 |

---

## 주요 기능

### 1. 가격 차트 (봉 단위)
- **캔들스틱 + 거래량 막대** + 전일 종가 기준선, **원화 병기**
- 탭 = **봉 단위**: 1일(일봉) · 1주(주봉) · 1달(월봉) · 3달(분기봉) · 1년(연봉)
- 각 탭은 **최근 구간부터** 표시, 왼쪽으로 드래그해 과거 봉 조회
- 가격 폭이 큰 장기 차트는 **로그 스케일** 자동 적용
- 조작: 드래그=이동 · 스크롤=확대/축소 · 더블클릭=원상복구

### 2. 감정분류 — 두 엔진 (sentiment.py)
`SENTIMENT_ENGINE` 환경변수로 전환 (`finbert` | `sklearn`).

| 엔진 | 설명 | 정확도 |
|---|---|---|
| **FinBERT** | 금융 특화 사전학습 BERT (ProsusAI/finbert) | 약 88~90% |
| **자체 학습 모델** | TF-IDF(단어+문자 n-gram) + LinearSVC — **직접 학습** | 77.9% |

- 심리점수 = 각 뉴스의 `P(긍정) - P(부정)` 평균을 0~100으로 변환 (50=중립)

### 3. 준실시간 시세 + 투자 목표 분석
- yfinance로 현재가·당일 고저·전일종가 조회 (**약 15분 지연**, 무료) → 시나리오에 실제 가격 주입
- **기간별 목표 수익** 입력(1주/1달/3달/6달/1년 또는 개월 직접입력)
- 종목의 **1년 수익률·연변동성·최대낙폭** 통계로 **목표 현실성 분석**
- 무리한 목표면 **"기간을 늘리거나 목표를 낮추라"**는 구체적 대안 추천

---

## 자체 학습 모델 상세 (발표용)

- **데이터**: Financial PhraseBank (Malo et al., 2014). `Sentences_50Agree.txt` 4,846문장
- **방식**: TF-IDF(단어 1~2gram + 문자 3~5gram) → 로지스틱회귀/선형SVM/나이브베이즈 비교 → **LinearSVC**

### 데이터 동의율별 정확도 (같은 파이프라인, DATA_FILE만 변경)

| 데이터셋 | 문장수 | 정확도 | macro-F1 |
|---|---|---|---|
| 50Agree (기본) | 4,846 | 77.9% | 0.729 |
| 66Agree | 4,217 | 81.9% | 0.772 |
| 75Agree | 3,453 | 86.8% | 0.817 |
| AllAgree (만장일치) | 2,264 | 92.1% | 0.888 |

> 50Agree는 주석자 절반만 동의한 **가장 노이즈 많은** 세트라 상한이 낮음 → "왜 77%냐"에 데이터 품질로 답변 가능.
> 정밀도는 BERT 계열이 우위 → 앱 기본 엔진 FinBERT, 자체 모델은 비교·폴백·배포용.

---

## API 엔드포인트 (main.py)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 헬스체크 |
| POST | `/analyze` | 심리분석만 |
| GET | `/price/{ticker}?period=1달` | 가격 차트 (OHLCV + 통계 + 환율) |
| GET | `/quote/{ticker}` | 준실시간 시세 |
| POST | `/generate` | 뉴스심리 + 시세 + 통계 + 목표 → GPT-5.4 시나리오 |

---

## 환경변수 (.env)

```ini
NEWSAPI_KEY=...            # newsapi.org 무료키
ELICE_API_KEY=...          # 엘리스 AI 클라우드 (OpenAI 호환 프록시)
ELICE_BASE_URL=...
ELICE_MODEL=openai/gpt-5.4
SENTIMENT_ENGINE=finbert   # finbert(로컬) | sklearn(배포 기본)
```

> `.env`는 `.gitignore`로 제외되어 깃허브에 올라가지 않음.

---

> ⚠️ 교육·시뮬레이션 목적의 예시이며 실제 투자 조언이 아님.
