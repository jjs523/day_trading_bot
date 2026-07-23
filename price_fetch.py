"""
price_fetch.py
---------------
야후 파이낸스(yfinance)로 티커의 가격 차트 데이터를 가져오는 모듈.
버튼 = '봉 단위'. 각 탭은 그 단위의 캔들을 상장 이후(또는 최근 1년)만큼 보여줘서
스크롤로 이전 캔들까지 볼 수 있다.
  1일 = 일봉(최근 1년) · 1달 = 월봉(전체) · 3달 = 분기봉(전체) · 1년 = 연봉(전체)
OHLCV(시가/고가/저가/종가/거래량) + 전일대비 등락 + 환율 반환.
"""

import yfinance as yf

# 기간 라벨 -> (yfinance period, interval).  '봉 단위' 개념.
PERIODS = {
    "1일": ("10y", "1d"),    # 일봉: 최근 10년
    "1주": ("max", "1wk"),   # 주봉: 상장 이후 전체
    "1달": ("max", "1mo"),   # 월봉: 전체
    "3달": ("max", "3mo"),   # 분기봉: 전체
    "1년": ("max", "1mo"),   # 연봉: 월봉을 연 단위로 리샘플 (yfinance에 연봉 간격 없음)
}
# 연봉처럼 yfinance가 직접 지원 안 하는 단위는 리샘플. (pandas 규칙: YS=연초 기준)
RESAMPLE = {"1년": "YS"}


def fetch_price_chart(ticker: str, period_label: str = "1달") -> dict:
    """
    ticker: 종목 티커 (예: TSLA)
    period_label: "1일" | "1달" | "3달" | "1년"  (각각 일/월/분기/연 봉)

    반환:
      dates / open / high / low / close / volume: 각 캔들의 OHLCV
      prices: close 별칭 · current_price / prev_close / change_amt / change_pct
      krw_rate: 원화 환산용 환율
    """
    if period_label not in PERIODS:
        raise ValueError(f"알 수 없는 기간: {period_label}")

    period, interval = PERIODS[period_label]
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)

    if hist.empty:
        raise RuntimeError(f"'{ticker}' 가격 데이터를 가져오지 못했습니다.")

    # 연봉 등: 지원 안 되는 단위는 OHLCV 규칙으로 리샘플
    rule = RESAMPLE.get(period_label)
    if rule:
        hist = hist.resample(rule).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])

    def col(name):
        return [round(float(v), 2) for v in hist[name]]

    dates = [d.isoformat() for d in hist.index]
    close = col("Close")
    volume = [int(v) for v in hist["Volume"]]

    # 전일대비: 일봉 데이터에서 마지막 두 종가로 계산 (인트라데이 노이즈 방지)
    daily = t.history(period="5d", interval="1d")
    if len(daily) >= 2:
        prev_close = round(float(daily["Close"].iloc[-2]), 2)
        current_price = round(float(daily["Close"].iloc[-1]), 2)
    else:
        prev_close = close[0]
        current_price = close[-1]

    change_amt = round(current_price - prev_close, 2)
    change_pct = round((change_amt / prev_close) * 100, 2) if prev_close else 0.0

    return {
        "ticker": ticker.upper(),
        "period": period_label,
        "interval": interval,
        "dates": dates,
        "open": col("Open"),
        "high": col("High"),
        "low": col("Low"),
        "close": close,
        "prices": close,          # 라인 차트 호환용 별칭
        "volume": volume,
        "current_price": current_price,
        "prev_close": prev_close,
        "change_amt": change_amt,
        "change_pct": change_pct,
        "krw_rate": usd_to_krw_rate(),          # USD→KRW 환율 (원화 표시용)
    }


def price_stats(ticker: str) -> dict | None:
    """
    목표 달성 가능성 판단용 통계 (최근 1년 일봉 기준).
      return_1y_pct        : 최근 1년 총수익률(%)
      volatility_annual_pct: 연환산 변동성(%)  (일간수익률 표준편차 × √252)
      max_drawdown_1y_pct  : 최근 1년 최대 낙폭(%)
      avg_monthly_return_pct: 월평균 수익률(%) (1년 수익률/12, 대략치)
    """
    try:
        h = yf.Ticker(ticker).history(period="1y", interval="1d")
        if h.empty or len(h) < 20:
            return None
        closes = h["Close"]
        ret_1y = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
        daily_ret = closes.pct_change().dropna()
        vol_ann = float(daily_ret.std() * (252 ** 0.5) * 100)
        max_dd = float(((closes - closes.cummax()) / closes.cummax()).min() * 100)
        return {
            "return_1y_pct": round(float(ret_1y), 1),
            "volatility_annual_pct": round(vol_ann, 1),
            "max_drawdown_1y_pct": round(max_dd, 1),
            "avg_monthly_return_pct": round(float(ret_1y) / 12, 1),
        }
    except Exception:
        return None


def usd_to_krw_rate() -> float:
    """USD→KRW 환율. yfinance 'KRW=X'. 실패 시 대략값(1,380)으로 폴백."""
    try:
        fx = yf.Ticker("KRW=X")
        rate = None
        try:
            fi = fx.fast_info
            rate = fi.get("lastPrice") or fi.get("last_price")
        except Exception:
            pass
        if not rate:
            h = fx.history(period="5d", interval="1d")
            if not h.empty:
                rate = float(h["Close"].iloc[-1])
        return round(float(rate), 2) if rate else 1380.0
    except Exception:
        return 1380.0


def get_quote(ticker: str) -> dict:
    """
    준(準)실시간 시세. yfinance 무료 데이터는 약 15분 지연이라 '실시간'은 아니지만,
    데이 트레이딩 시나리오에 쓸 구체적 가격대(현재가·당일 고저·전일종가)를 제공.
    """
    t = yf.Ticker(ticker)

    # fast_info: 가장 가벼운 최신 시세 경로
    price = day_high = day_low = day_open = prev_close = None
    try:
        fi = t.fast_info
        price = fi.get("lastPrice") or fi.get("last_price")
        day_high = fi.get("dayHigh") or fi.get("day_high")
        day_low = fi.get("dayLow") or fi.get("day_low")
        day_open = fi.get("open")
        prev_close = fi.get("previousClose") or fi.get("previous_close")
    except Exception:
        pass

    # 보강: 일봉으로 전일종가/당일가 채우기
    if price is None or prev_close is None:
        daily = t.history(period="5d", interval="1d")
        if not daily.empty:
            price = price or float(daily["Close"].iloc[-1])
            if prev_close is None and len(daily) >= 2:
                prev_close = float(daily["Close"].iloc[-2])
            day_high = day_high or float(daily["High"].iloc[-1])
            day_low = day_low or float(daily["Low"].iloc[-1])
            day_open = day_open or float(daily["Open"].iloc[-1])

    def r(x):
        return round(float(x), 2) if x is not None else None

    price, prev_close = r(price), r(prev_close)
    change_amt = round(price - prev_close, 2) if (price and prev_close) else None
    change_pct = round(change_amt / prev_close * 100, 2) if (change_amt and prev_close) else None

    return {
        "ticker": ticker.upper(),
        "price": price,
        "day_open": r(day_open),
        "day_high": r(day_high),
        "day_low": r(day_low),
        "prev_close": prev_close,
        "change_amt": change_amt,
        "change_pct": change_pct,
        "krw_rate": usd_to_krw_rate(),          # USD→KRW 환율
        "delay_note": "약 15분 지연 시세 (yfinance 무료)",
    }


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    print("[quote]", get_quote(tk))
    for label in PERIODS:
        r = fetch_price_chart(tk, label)
        print(f"[{label}] {r['current_price']} ({r['change_amt']:+} / {r['change_pct']:+}%) "
              f"- {len(r['close'])}개 캔들, 거래량 최대 {max(r['volume']):,}")
