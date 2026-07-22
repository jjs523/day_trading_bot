"""
news_fetch.py
-------------
NewsAPI(newsapi.org)로 종목 뉴스를 수집해서 감정분석에 바로 넣을 형태로 정리.
"""

import os
import requests
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ENDPOINT = "https://newsapi.org/v2/everything"

# 신뢰할 만한 금융/경제 매체. 여기로 제한하면 가십·오피니언이 걸러져 심리 신호가 깨끗해짐.
FINANCE_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,marketwatch.com,finance.yahoo.com,"
    "wsj.com,investing.com,fool.com,benzinga.com,seekingalpha.com,"
    "businessinsider.com,forbes.com,barrons.com,thestreet.com,ft.com"
)

# 티커보다 회사명으로 검색해야 뉴스가 잘 잡힘. 자주 쓰는 것만 매핑.
TICKER_TO_QUERY = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia", "MSFT": "Microsoft",
    "AMZN": "Amazon", "GOOGL": "Google", "GOOG": "Google", "META": "Meta",
    "AMD": "AMD", "INTC": "Intel", "NFLX": "Netflix", "BABA": "Alibaba",
    "COIN": "Coinbase", "PLTR": "Palantir", "UBER": "Uber",
}


def _is_junk(text: str) -> bool:
    """비영어(일본어·기타)나 라틴문자 비율이 낮은 항목 제거."""
    if not text:
        return True
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return ascii_letters < len(text) * 0.5


def _query(key: str, params: dict) -> list:
    resp = requests.get(ENDPOINT, params=params,
                        headers={"X-Api-Key": key}, timeout=15)
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI 오류 [{data.get('code')}]: {data.get('message')}")
    return data.get("articles", [])


def fetch_news(query: str, language: str = "en", page_size: int = 20,
               days: int = 7, api_key: str | None = None,
               finance_only: bool = True) -> list[dict]:
    """
    query: 티커(AAPL) 또는 회사명/키워드(Apple). 티커면 회사명으로 자동 변환.
    finance_only: True면 금융 매체 우선. 결과가 적으면 자동으로 전체 매체로 폴백.
    """
    key = api_key or NEWSAPI_KEY
    if not key:
        raise RuntimeError(
            "NEWSAPI_KEY가 없어. .env 파일에 NEWSAPI_KEY=너의키 를 추가해줘.")

    q = TICKER_TO_QUERY.get(query.strip().upper(), query.strip())
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    base = {
        "q": q,
        "language": language,
        "sortBy": "publishedAt",      # 최신순
        "pageSize": page_size,
        "from": from_date,
    }

    # 관련성 높은 순서로 시도하다가 5건 이상 나오면 채택 (없으면 점점 완화)
    #  1) 금융매체 + 제목/요약에 회사명       ← 가장 깨끗
    #  2) 제목/요약에 회사명 (전체 매체)
    #  3) 본문까지 포함 (최후)
    attempts = []
    if finance_only:
        attempts.append(dict(base, domains=FINANCE_DOMAINS, searchIn="title,description"))
    attempts.append(dict(base, searchIn="title,description"))
    attempts.append(dict(base))

    best = []
    for params in attempts:
        arts = _normalize(_query(key, params))
        if len(arts) >= 5:
            return arts
        if len(arts) > len(best):
            best = arts
    return best


def _normalize(articles: list[dict]) -> list[dict]:
    """감정분석에 쓸 필드만 추려서 정리. text = 제목(+요약). 비영어/쓰레기 제거."""
    out = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        desc = (a.get("description") or "").strip()
        text = title if not desc else f"{title}. {desc}"
        if _is_junk(text):          # 일본어 등 비영어·깨진 항목 스킵
            continue
        out.append({
            "title": title,
            "description": desc,
            "text": text,
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
        })
    return out


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    news = fetch_news(ticker, page_size=10, days=7)
    print(f"'{ticker}' 뉴스 {len(news)}건 수집\n")
    for n in news[:10]:
        print(f"- [{n['source']}] {n['title']}")