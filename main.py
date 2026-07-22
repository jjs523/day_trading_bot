"""
FastAPI 백엔드 - 뉴스 심리 기반 데이 트레이딩 시나리오 봇
흐름:  티커 -> 뉴스 수집(NewsAPI) -> 감정분류 ML(심리점수) -> GPT-5 mini 시나리오

엘리스 AI 클라우드(OpenAI 호환 프록시)로 GPT-5 mini 호출.
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

load_dotenv()

ELICE_API_KEY = os.getenv("ELICE_API_KEY")
ELICE_BASE_URL = os.getenv("ELICE_BASE_URL")
ELICE_MODEL = os.getenv("ELICE_MODEL", "openai/gpt-5-mini")

if not ELICE_API_KEY or not ELICE_BASE_URL:
    print("[경고] .env 에 ELICE_API_KEY 또는 ELICE_BASE_URL 이 없습니다.")
if not os.getenv("NEWSAPI_KEY"):
    print("[경고] .env 에 NEWSAPI_KEY 가 없습니다. 뉴스 심리분석이 동작하지 않습니다.")

client = OpenAI(api_key=ELICE_API_KEY, base_url=ELICE_BASE_URL)

app = FastAPI(title="뉴스 심리 기반 트레이딩 시나리오 봇")


class TradeRequest(BaseModel):
    seed_money: int = Field(..., description="시드 머니 (원)")
    target_profit: int = Field(..., description="일일 목표 수익금 (원)")
    ticker: str = Field(..., description="관심 티커 (예: TSLA)")
    trade_type: str = Field("데이 트레이딩", description="매매 방식")
    days: int = Field(7, description="뉴스 조회 기간(일)")
    max_news: int = Field(20, description="분석할 최대 뉴스 수")


SYSTEM_PROMPT = """당신은 미국 주식 시장의 단기 매매를 분석하는 퀀트 트레이딩 교육 어시스턴트입니다.
사용자 조건과 '직접 학습한 감정분류 ML 모델이 분석한 뉴스 심리 결과'가 함께 주어집니다.
감정적 표현을 배제하고, 뉴스 심리와 수치·확률에 기반해 객관적인 어조로 아래 형식에 맞춰 작성하세요.

1. 뉴스 심리 요약: 주어진 심리점수/분포가 매매에 시사하는 바 (긍정/부정/혼조 해석)
2. 변동성 분석: 해당 티커의 일반적인 단기 변동성 특성
3. 타점 시나리오: 분할 매수 진입 구간과 1·2차 청산 목표가 (시드머니·목표수익 기준)
4. 리스크 관리: 손절매(Stop-loss) 기준, 오버나잇의 경우 프리/애프터마켓 리스크

중요:
- 심리점수가 혼조(40~60)일 때는 불확실성을 명확히 언급하고 보수적으로 접근하세요.
- 이것은 교육 및 시뮬레이션 목적의 예시이며 실제 투자 조언이 아닙니다.
- 답변 마지막에 "※ 본 내용은 교육용 예시이며 실제 투자 판단의 근거로 사용할 수 없습니다." 문구를 반드시 포함하세요.
- 실시간 시세를 모른다는 점을 전제로, 일반적 매매 전략의 틀을 설명하세요."""


def build_user_prompt(req: TradeRequest, senti: dict | None, news_err: str | None) -> str:
    if senti and senti.get("n"):
        counts = ", ".join(f"{k} {v}건" for k, v in senti["counts"].items())
        headlines = "\n".join(
            f"  - [{it['label_ko']}] {it['text'][:90]}" for it in senti["items"][:8]
        )
        senti_block = (
            f"[뉴스 심리 분석 결과] (직접 학습한 감정분류 ML, 정확도 약 77.9%)\n"
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
        f"[사용자 조건]\n"
        f"  시드머니 {req.seed_money:,}원 / 일일 목표수익 {req.target_profit:,}원\n"
        f"  티커 {req.ticker} / 매매방식 {req.trade_type}\n\n"
        f"{senti_block}\n\n"
        f"위 뉴스 심리를 반영해 단기 매매 시나리오를 작성해주세요."
    )


def get_sentiment(req: TradeRequest):
    """뉴스 수집 + 감정분석. 실패해도 서버가 죽지 않도록 예외를 잡아 반환."""
    try:
        news = fetch_news(req.ticker, page_size=req.max_news, days=req.days)
        senti = analyze_news([n["text"] for n in news])
        for item, art in zip(senti["items"], news):
            item["source"] = art.get("source")
            item["url"] = art.get("url")
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


@app.post("/generate")
def generate_scenario(req: TradeRequest):
    """뉴스 심리 + 사용자 조건 -> GPT-5 mini 시나리오 (비스트리밍)."""
    senti, news_err = get_sentiment(req)
    try:
        response = client.chat.completions.create(
            model=ELICE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(req, senti, news_err)},
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
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")