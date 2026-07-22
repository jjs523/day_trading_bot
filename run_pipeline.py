"""
run_pipeline.py
---------------
전체 흐름 데모:  종목 입력 -> 뉴스 수집 -> 감정분석(심리점수) -> LLM 프롬프트 생성

이 파일이 앞단(뉴스+ML)과 뒷단(트레이딩 봇 LLM)을 잇는 접착제.
build_llm_prompt()가 만든 프롬프트를 기존 GPT-5 mini 호출부에 넣으면 시나리오가 나옴.

실행:
  python run_pipeline.py TSLA --seed 1000000 --target 30000        # 실제 뉴스 (키 필요)
  python run_pipeline.py TSLA --seed 1000000 --target 30000 --mock # 오프라인 테스트
"""

import argparse
from sentiment import analyze_news

# ── 오프라인 테스트용 가짜 뉴스 (실제 미국주식 헤드라인 스타일) ──────────────
MOCK_ARTICLES = [
    {"source": "Reuters", "text": "Tesla shares jump after the company reports record quarterly deliveries."},
    {"source": "Bloomberg", "text": "Tesla warns of softer demand as price cuts squeeze profit margins."},
    {"source": "CNBC", "text": "Analysts remain divided on Tesla's outlook heading into the next quarter."},
    {"source": "WSJ", "text": "Tesla unveils a new battery technology aimed at cutting production costs."},
    {"source": "AP", "text": "Regulators open a probe into Tesla's driver-assistance system."},
    {"source": "MarketWatch", "text": "Tesla stock rises as investors cheer stronger-than-expected margins."},
]


def build_llm_prompt(ticker: str, seed: int, target: int, senti: dict) -> str:
    """감정분석 결과 + 사용자 조건을 LLM 프롬프트로 조립."""
    counts = ", ".join(f"{k} {v}건" for k, v in senti["counts"].items())
    headlines = "\n".join(
        f"  - [{it['label_ko']}] {it['text'][:80]}" for it in senti["items"][:8]
    )
    return f"""너는 신중한 트레이딩 어시스턴트다. 아래 정보를 바탕으로 트레이딩 시나리오를 제시하라.

[종목] {ticker}
[사용자 조건] 시드머니 {seed:,}원, 일일 목표수익 {target:,}원
[뉴스 심리 분석 결과] (직접 학습한 감정분류 모델 기준)
  - 종합 심리점수: {senti['sentiment_score']}/100 ({senti['verdict']})
  - 뉴스 분포: {counts} (총 {senti['n']}건)
  - 주요 헤드라인:
{headlines}

[요구사항]
1. 위 뉴스 심리를 반영한 매매 관점(매수/관망/매도)을 먼저 제시.
2. 구체적 진입 타점과 손절매(Stop-loss) 기준을 제시.
3. 시드머니·목표수익에 비춘 포지션 크기와 리스크를 언급.
4. 뉴스 심리가 혼조일 경우 그 불확실성을 명시.
* 투자 조언이 아닌 참고용 시나리오임을 마지막에 한 줄로 밝혀라."""


def run(ticker: str, seed: int, target: int, days: int = 7,
        max_news: int = 20, mock: bool = False):
    if mock:
        articles = MOCK_ARTICLES
        print(f"[MOCK] 가짜 뉴스 {len(articles)}건으로 테스트\n")
    else:
        from news_fetch import fetch_news
        articles = fetch_news(ticker, page_size=max_news, days=days)
        print(f"[NewsAPI] '{ticker}' 뉴스 {len(articles)}건 수집\n")

    texts = [a["text"] for a in articles]
    senti = analyze_news(texts)

    print(f"종합 심리점수: {senti['sentiment_score']}/100 ({senti['verdict']})")
    print(f"분포: {senti['counts']}\n")
    for a, it in zip(articles, senti["items"]):
        print(f"  [{it['label_ko']} {it['confidence']*100:.0f}%] "
              f"({a.get('source','?')}) {it['text'][:70]}")

    prompt = build_llm_prompt(ticker, seed, target, senti)
    print("\n" + "=" * 60)
    print("아래 프롬프트를 기존 GPT-5 mini 호출부에 넣으면 시나리오 생성됨:")
    print("=" * 60)
    print(prompt)
    return senti, prompt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="TSLA")
    ap.add_argument("--seed", type=int, default=1_000_000)
    ap.add_argument("--target", type=int, default=30_000)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-news", type=int, default=20)
    ap.add_argument("--mock", action="store_true", help="뉴스 API 없이 오프라인 테스트")
    args = ap.parse_args()
    run(args.ticker, args.seed, args.target, args.days, args.max_news, args.mock)
