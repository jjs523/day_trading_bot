"""
FastAPI 백엔드 - 뉴스 심리 기반 데이 트레이딩 시나리오 봇
흐름:  티커 -> 뉴스 수집(NewsAPI) -> 감정분류 ML(심리점수) -> GPT-5.4 시나리오
       + 가격 차트(1일/1달/3달/1년, 전일대비)는 /price 에서 별도 제공

엘리스 AI 클라우드(OpenAI 호환 프록시)로 GPT-5.4 호출.
.env 에 필요한 값: ELICE_API_KEY, ELICE_BASE_URL, ELICE_MODEL, NEWSAPI_KEY
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# 우리가 만든 모듈
from news_fetch import fetch_news
from sentiment import analyze_news
from price_fetch import fetch_price_chart, get_quote, price_stats, PERIODS

load_dotenv()

ELICE_API_KEY = os.getenv("ELICE_API_KEY")
ELICE_BASE_URL = os.getenv("ELICE_BASE_URL")
ELICE_MODEL = os.getenv("ELICE_MODEL", "openai/gpt-5.4")

if not ELICE_API_KEY or not ELICE_BASE_URL:
    print("[경고] .env 에 ELICE_API_KEY 또는 ELICE_BASE_URL 이 없습니다.")
if not os.getenv("NEWSAPI_KEY"):
    print("[경고] .env 에 NEWSAPI_KEY 가 없습니다. 뉴스 심리분석이 동작하지 않습니다.")

client = OpenAI(api_key=ELICE_API_KEY, base_url=ELICE_BASE_URL)

app = FastAPI(title="뉴스 심리 기반 트레이딩 시나리오 봇")


class TradeRequest(BaseModel):
    seed_money: int = Field(..., description="시드 머니 (원)")
    target_profit: int = Field(..., description="목표 기간 동안 원하는 수익금 (원)")
    ticker: str = Field(..., description="관심 티커 (예: TSLA)")
    target_period_label: str = Field("1달", description="목표 기간 라벨 (예: 1달, 3개월)")
    target_period_days: int = Field(30, description="목표 기간(일 단위)")
    days: int = Field(7, description="뉴스 조회 기간(일)")
    max_news: int = Field(20, description="분석할 최대 뉴스 수")


SYSTEM_PROMPT = """당신은 미국 주식 '목표 수익 달성 가능성'을 분석하는 퀀트 투자 교육 어시스턴트입니다.
사용자는 '시드머니', '목표 기간', '그 기간 동안 원하는 수익금'을 제시합니다.
여기에 '뉴스 심리 분석 결과', '준실시간 시세', '해당 종목의 최근 수익률·변동성 통계'가 함께 주어집니다.
감정적 표현을 배제하고 수치·확률에 기반해 객관적으로 아래 형식으로 작성하세요.

1. 목표 현실성 분석: 목표 수익률(= 원하는 수익금 / 시드머니)을 '목표 기간' 안에 달성 가능한지,
   종목의 연수익률·연변동성·최대낙폭 통계에 비추어 평가하세요. 무리라면 그 이유를 수치로 제시.
2. 기간·목표 추천: 현재 목표가 비현실적이면 구체적 대안을 제시하세요.
   예: "1달로는 어려우니 3달로 늘리면 통계적으로 더 현실적", 또는 "목표를 X원으로 낮추면 달성 확률↑".
   현실적이면 그대로 진행해도 되는 근거를 제시.
3. 뉴스 심리 반영: 주어진 심리점수/분포가 매매 타이밍에 시사하는 바.
4. 매매 전략: 준실시간 시세를 기준으로 분할 매수 진입 구간·1·2차 목표가·손절가를 '실제 숫자'로 제시.
5. 리스크 관리: 손절 기준, 하락장·변동성 리스크.

중요:
- 목표 수익률과 기간을 항상 함께 언급하고, 종목 통계와 직접 비교해 현실성을 판단하세요.
- 심리점수가 혼조(40에서 60)면 불확실성을 명확히 언급하고 보수적으로 접근하세요.
- 시세 정보가 있으면 현재가/당일 고저/전일 종가 기준으로 구체적 숫자를 쓰세요(약 15분 지연 시세임을 한 번 밝힘).
- 이것은 교육 및 시뮬레이션 목적의 예시이며 실제 투자 조언이 아닙니다.
- 답변 마지막에 "※ 본 내용은 교육용 예시이며 실제 투자 판단의 근거로 사용할 수 없습니다." 문구를 반드시 포함하세요.

[표기 규칙 - 반드시 지킬 것]
- 물결표(~)를 절대 쓰지 마세요. 범위는 '에서' 또는 붙임표(-)로 쓰세요. 예: "0.8~1.5%"(X) -> "0.8%에서 1.5%"(O)
- 주요 가격(진입가·목표가·손절가)은 달러와 원화를 괄호로 병기하세요."""


def build_quote_block(quote: dict | None) -> str:
    if not quote or quote.get("price") is None:
        return "[현재 시세] 조회 실패 - 구체적 가격 없이 일반 전략 틀로 설명하세요."
    chg = ""
    if quote.get("change_amt") is not None:
        chg = f" (전일대비 {quote['change_amt']:+}, {quote['change_pct']:+}%)"
    rate = quote.get("krw_rate")
    fx = f"\n  - 환율(USD/KRW): {rate:,.0f}원" if rate else ""
    won_rule = (
        "\n  주요 가격(진입가·목표가·손절가)은 달러와 함께 원화 환산도 괄호로 병기하세요. "
        f"예: $325.89(약 {round(325.89*rate):,}원)" if rate else ""
    )
    return (
        f"[현재 시세] {quote['delay_note']}\n"
        f"  - 현재가: ${quote['price']}{chg}\n"
        f"  - 당일 시가 ${quote.get('day_open')} / 고가 ${quote.get('day_high')} / 저가 ${quote.get('day_low')}\n"
        f"  - 전일 종가: ${quote.get('prev_close')}{fx}\n"
        f"  위 실제 가격을 기준으로 진입가·목표가·손절가를 숫자로 제시하세요.{won_rule}"
    )


def build_target_block(req: TradeRequest) -> str:
    """목표 기간 + 목표 수익 + 목표 수익률(=수익/시드) 요약."""
    target_pct = (req.target_profit / req.seed_money * 100) if req.seed_money else 0.0
    return (
        f"[투자 목표]\n"
        f"  - 시드머니: {req.seed_money:,}원\n"
        f"  - 목표 기간: {req.target_period_label} (약 {req.target_period_days}일)\n"
        f"  - 목표 수익금: {req.target_profit:,}원\n"
        f"  - 목표 수익률: {target_pct:.1f}% (목표수익 / 시드머니, 해당 기간 동안)\n"
        f"  이 목표 수익률을 목표 기간 안에 달성 가능한지 종목 통계와 비교해 평가하고, "
        f"무리라면 기간을 늘리거나 목표를 낮추는 구체적 대안을 제시하세요."
    )


def build_stats_block(stats: dict | None) -> str:
    if not stats:
        return "[종목 통계] 조회 실패 - 일반적 시장 통념으로 현실성을 판단하세요."
    return (
        f"[종목 통계] (최근 1년, 목표 현실성 판단 근거)\n"
        f"  - 최근 1년 총수익률: {stats['return_1y_pct']:+}%\n"
        f"  - 연환산 변동성: {stats['volatility_annual_pct']}%\n"
        f"  - 최근 1년 최대 낙폭(MDD): {stats['max_drawdown_1y_pct']}%\n"
        f"  - 월평균 수익률(대략): {stats['avg_monthly_return_pct']:+}%"
    )


def build_user_prompt(req: TradeRequest, senti: dict | None, news_err: str | None,
                      quote: dict | None = None, stats: dict | None = None) -> str:
    if senti and senti.get("n"):
        counts = ", ".join(f"{k} {v}건" for k, v in senti["counts"].items())
        headlines = "\n".join(
            f"  - [{it['label_ko']}] {it['text'][:90]}" for it in senti["items"][:8]
        )
        eng = senti.get("engine_name", "감정분류 ML")
        acc = senti.get("engine_accuracy", "")
        senti_block = (
            f"[뉴스 심리 분석 결과] ({eng}, 정확도 {acc})\n"
            f"  - 종합 심리점수: {senti['sentiment_score']}/100 ({senti['verdict']})\n"
            f"  - 뉴스 분포: {counts} (총 {senti['n']}건)\n"
            f"  - 주요 헤드라인:\n{headlines}"
        )
    else:
        reason = f" (사유: {news_err})" if news_err else ""
        senti_block = (
            f"[뉴스 심리 분석 결과] 뉴스를 가져오지 못했습니다{reason}.\n"
            f"  - 심리 정보 없이 일반적인 매매 전략의 틀만 설명하세요."
        )

    return (
        f"[종목] {req.ticker}\n\n"
        f"{build_target_block(req)}\n\n"
        f"{build_stats_block(stats)}\n\n"
        f"{build_quote_block(quote)}\n\n"
        f"{senti_block}\n\n"
        f"위 투자 목표의 현실성을 먼저 판단하고(필요하면 기간·목표 조정 추천), "
        f"실제 시세와 뉴스 심리를 반영해 매매 시나리오를 작성해주세요."
    )


def get_sentiment(req: TradeRequest):
    """뉴스 수집 + 감정분석. 실패해도 서버가 죽지 않도록 예외를 잡아 반환."""
    try:
        news = fetch_news(req.ticker, page_size=req.max_news, days=req.days)
        senti = analyze_news([n["text"] for n in news])
        for item, art in zip(senti["items"], news):
            item["source"] = art.get("source")
            item["url"] = art.get("url")
            item["publishedAt"] = art.get("publishedAt")
        return senti, None
    except Exception as e:
        print(f"[뉴스 수집 실패] {e}")
        return None, str(e)


@app.get("/")
def health():
    return {"status": "ok", "model": ELICE_MODEL}


@app.post("/analyze")
def analyze_only(req: TradeRequest):
    """심리분석만 (UI에서 시나리오 생성 전 미리보기용)."""
    senti, news_err = get_sentiment(req)
    return {"sentiment": senti, "news_error": news_err}


@app.get("/price/{ticker}")
def get_price(ticker: str, period: str = "1달"):
    """가격 차트 데이터 (1일 / 1달 / 3달 / 1년) + 전일대비 등락."""
    if period not in PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"period 는 {list(PERIODS.keys())} 중 하나여야 합니다.",
        )
    try:
        return fetch_price_chart(ticker, period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"가격 데이터 조회 실패: {e}")


@app.get("/quote/{ticker}")
def quote(ticker: str):
    """준실시간(약 15분 지연) 현재가·당일 고저·전일종가."""
    try:
        return get_quote(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"시세 조회 실패: {e}")


def _safe_quote(ticker: str):
    try:
        return get_quote(ticker)
    except Exception as e:
        print(f"[시세 조회 실패] {e}")
        return None


def _safe_stats(ticker: str):
    try:
        return price_stats(ticker)
    except Exception as e:
        print(f"[통계 조회 실패] {e}")
        return None


@app.post("/generate")
def generate_scenario(req: TradeRequest):
    """뉴스 심리 + 준실시간 시세 + 종목 통계 + 투자 목표 -> GPT-5.4 시나리오."""
    senti, news_err = get_sentiment(req)
    quote_data = _safe_quote(req.ticker)
    stats = _safe_stats(req.ticker)
    try:
        response = client.chat.completions.create(
            model=ELICE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_user_prompt(req, senti, news_err, quote_data, stats)},
            ],
            max_completion_tokens=8000,
        )
        content = response.choices[0].message.content

        if not content:
            print("=== 빈 응답 디버그 ===")
            print("finish_reason:", response.choices[0].finish_reason)
            print("전체 응답:", response)
            content = "(모델이 빈 응답을 반환했습니다. 터미널 로그를 확인하세요.)"

        return {
            "scenario": content,
            "sentiment": senti,
            "news_error": news_err,
            "quote": quote_data,
            "stats": stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")