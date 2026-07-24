"""
app.py - Streamlit UI
뉴스 심리 기반 트레이딩 시나리오 봇의 프런트엔드.
FastAPI(main.py)를 먼저 켜둔 상태에서 실행:
  터미널 1:  uvicorn main:app --reload
  터미널 2:  streamlit run app.py
"""
import math
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 기존 코드
# API_URL = "http://127.0.0.1:8000"

# 수정할 코드
API_URL = st.secrets.get("SERVER_BASE", "http://127.0.0.1:8000")

PERIOD_LABELS = ["1일", "1주", "1달", "3달", "1년"]

# 목표 기간 프리셋 -> 일수
TARGET_PERIODS = {"1주": 7, "1달": 30, "3달": 90, "6달": 180, "1년": 365, "직접 입력(개월)": None}

# 차트 탭별 '기본으로 보여줄 최근 구간'(일수). None = 전체. 이전 봉은 왼쪽으로 스크롤.
DEFAULT_VIEW_DAYS = {
    "1일": 120,    # 일봉: 최근 4개월 (오늘이 오른쪽 끝)
    "1주": 300,    # 주봉: 최근 약 10개월
    "1달": 900,    # 월봉: 최근 약 2.5년
    "3달": 2000,   # 분기봉: 최근 약 5.5년
    "1년": None,   # 연봉: 전체 (봉이 몇 개 안 됨)
}

# 등락 색 (상승=초록, 하락=빨강). 미국장 관례.
UP_COLOR = "#16a34a"
DOWN_COLOR = "#dc2626"
FLAT_COLOR = "#64748b"

st.set_page_config(page_title="뉴스 심리 트레이딩 봇", page_icon="📈", layout="wide")
st.title("📈 뉴스 심리 기반 트레이딩 시나리오")
st.caption("금융 특화 FinBERT(+ 자체 학습 모델 비교)가 종목 뉴스 심리를 분석하고, "
           "준실시간 시세와 함께 GPT-5.4가 반영해 매매 시나리오를 생성합니다.")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("관심 티커", value="TSLA").strip().upper()
    seed_money = st.number_input("시드머니 (원)", min_value=0, value=1_000_000, step=100_000)
with col2:
    period_choice = st.selectbox("목표 기간", list(TARGET_PERIODS.keys()), index=1)  # 기본 1달
    if TARGET_PERIODS[period_choice] is None:
        months = st.number_input("몇 개월?", min_value=1, max_value=120, value=2, step=1)
        target_period_label = f"{int(months)}개월"
        target_period_days = int(months) * 30
    else:
        target_period_label = period_choice
        target_period_days = TARGET_PERIODS[period_choice]
    target_profit = st.number_input(
        f"목표 수익 (원) — {target_period_label} 동안", min_value=0, value=100_000, step=50_000)

# 목표 수익률 미리보기
if seed_money:
    _tp = target_profit / seed_money * 100
    st.caption(f"🎯 목표: **{target_period_label}** 동안 **{target_profit:,}원** "
               f"(목표 수익률 **{_tp:.1f}%**) · 시드 {seed_money:,}원")

with st.expander("고급 설정 (뉴스)"):
    days = st.slider("뉴스 조회 기간 (일)", 1, 30, 7)
    max_news = st.slider("분석할 뉴스 수", 5, 50, 20)

generate = st.button("🚀 시나리오 생성", type="primary", width="stretch")

# 줌: 드래그=이동, 스크롤=확대/축소, 더블클릭/오토스케일 버튼=원상복구
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


def _clean_scenario(text: str) -> str:
    """물결표(~)는 마크다운에서 취소선/아래첨자로 오해되므로 이스케이프."""
    return (text or "").replace("~", "\\~")


def _won(usd, rate) -> str:
    """달러 → 원화 환산 문자열. 예: '약 452,000원'."""
    if usd is None or not rate:
        return ""
    return f"약 {round(usd * rate):,}원"


# ── 가격 차트 ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price(ticker: str, period: str) -> dict:
    resp = requests.get(f"{API_URL}/price/{ticker}", params={"period": period}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def price_figure(data: dict) -> go.Figure:
    # DST 로 offset(-04:00/-05:00)이 섞이므로 UTC 파싱 후 미 동부시간으로 통일
    dates = pd.to_datetime(data["dates"], utc=True).tz_convert("America/New_York")
    closes, opens = data["close"], data["open"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )

    # ── 캔들스틱 (가격) ──
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=data["high"], low=data["low"], close=closes,
        increasing_line_color=UP_COLOR, increasing_fillcolor=UP_COLOR,
        decreasing_line_color=DOWN_COLOR, decreasing_fillcolor=DOWN_COLOR,
        line=dict(width=1), name="가격", showlegend=False,
    ), row=1, col=1)

    # 전일 종가 기준선
    fig.add_hline(y=data["prev_close"], line=dict(color=FLAT_COLOR, width=1, dash="dot"),
                  annotation_text="전일 종가", annotation_position="top left",
                  annotation_font_size=11, row=1, col=1)

    # ── 거래량 막대 (캔들 방향색) ──
    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(closes, opens)]
    fig.add_trace(go.Bar(
        x=dates, y=data["volume"], marker_color=vol_colors, marker_line_width=0,
        opacity=0.55, name="거래량", showlegend=False,
        hovertemplate="거래량 %{y:,}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,   # 캔들 기본 레인지슬라이더 제거
    )

    # 주말 갭 제거 (일봉일 때). rangebreaks 로 캔들이 빈틈없이 채워짐.
    rangebreaks = []
    if data.get("interval", "1d") == "1d":
        rangebreaks.append(dict(bounds=["sat", "mon"]))

    # 기본 뷰 = '최근' 구간만. (전체를 다 보여주면 봉이 너무 촘촘해 보기 힘듦)
    #  이전 봉은 왼쪽으로 드래그/스크롤해서 확인. 경계는 전체 데이터로 고정.
    lo, hi = dates.min(), dates.max()
    view_days = DEFAULT_VIEW_DAYS.get(data["period"])
    view_start = max(lo, hi - pd.Timedelta(days=view_days)) if view_days else lo
    fig.update_xaxes(showgrid=False, rangebreaks=rangebreaks,
                     range=[view_start, hi], minallowed=lo, maxallowed=hi)

    # y축은 '보이는 구간(최근)'의 가격에 맞춤 → 최근 캔들이 화면 가득 차게.
    vis = [i for i, d in enumerate(dates) if d >= view_start] or list(range(len(dates)))
    ylo = min(data["low"][i] for i in vis)
    yhi = max(data["high"][i] for i in vis)
    if ylo > 0 and yhi / ylo > 5:
        # 가격 폭이 매우 큰 구간(예: $1→$500)은 로그축이라야 초기 구간도 보임
        fig.update_yaxes(type="log", range=[math.log10(ylo * 0.9), math.log10(yhi * 1.1)],
                         tickprefix="$", gridcolor="rgba(148,163,184,0.18)",
                         side="right", row=1, col=1)
    else:
        ypad = (yhi - ylo) * 0.08 or (yhi * 0.02 if yhi else 1)
        fig.update_yaxes(range=[max(0, ylo - ypad), yhi + ypad], tickprefix="$",
                         gridcolor="rgba(148,163,184,0.18)", side="right", row=1, col=1)
    fig.update_yaxes(title_text="거래량", gridcolor="rgba(148,163,184,0.10)",
                     side="right", rangemode="tozero", row=2, col=1)  # 거래량 0부터
    return fig


def show_price(ticker: str):
    st.subheader(f"💹 {ticker} 가격")
    st.caption("드래그=이동 · 스크롤=확대/축소 · 더블클릭=원상복구")
    tabs = st.tabs(PERIOD_LABELS)
    for tab, period in zip(tabs, PERIOD_LABELS):
        with tab:
            try:
                data = fetch_price(ticker, period)
            except requests.exceptions.ConnectionError:
                st.error("백엔드에 연결 못 함. `uvicorn main:app --reload` 를 먼저 켜줘.")
                return
            except Exception as e:
                st.warning(f"가격 데이터를 못 가져왔어: {e}")
                continue

            amt, pct = data["change_amt"], data["change_pct"]
            rate = data.get("krw_rate")
            arrow = "▲" if amt > 0 else ("▼" if amt < 0 else "―")
            m1, m2 = st.columns([1, 4])
            with m1:
                st.metric(
                    "현재가", f"${data['current_price']:,.2f}",
                    f"{arrow} {abs(amt):,.2f} ({pct:+.2f}%)",
                    delta_color="normal",
                )
                st.caption(f"💱 {_won(data['current_price'], rate)}"
                           + (f"  (환율 {rate:,.0f}원)" if rate else ""))
                st.caption(f"전일 종가 ${data['prev_close']:,.2f} · {_won(data['prev_close'], rate)}")
            with m2:
                st.plotly_chart(price_figure(data), width="stretch",
                                config=PLOTLY_CONFIG)


# ── 심리 분석 ──────────────────────────────────────────────
def show_sentiment(senti: dict):
    score = senti["sentiment_score"]
    eng = senti.get("engine_name", "감정분류 ML")
    acc = senti.get("engine_accuracy", "")
    st.subheader("뉴스 심리 분석")
    st.caption(f"엔진: {eng} · 정확도 {acc}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("종합 심리점수", f"{score}/100", senti["verdict"])
    with c2:
        st.progress(int(score) / 100)
        counts = senti["counts"]
        df = pd.DataFrame(
            {"건수": [counts.get("부정", 0), counts.get("중립", 0), counts.get("긍정", 0)]},
            index=["부정", "중립", "긍정"],
        )
        st.bar_chart(df, height=180)

    with st.expander(f"분석한 뉴스 {senti['n']}건 보기"):
        for it in senti["items"]:
            src = it.get("source") or "?"
            url = it.get("url")
            title = it["text"][:100]
            line = f"**[{it['label_ko']} {int(it['confidence']*100)}%]** ({src}) {title}"
            st.markdown(f"- {line}" + (f" [🔗]({url})" if url else ""))


def show_quote(quote: dict):
    if not quote or quote.get("price") is None:
        return
    rate = quote.get("krw_rate")
    won = f" ({_won(quote['price'], rate)})" if rate else ""
    st.caption(f"💵 {quote['delay_note']} · 현재가 ${quote['price']}{won} · "
               f"당일 고 ${quote.get('day_high')} / 저 ${quote.get('day_low')} · "
               f"전일 종가 ${quote.get('prev_close')}")


def show_stats(stats: dict):
    """종목 통계 (목표 현실성 판단 근거)."""
    if not stats:
        return
    st.subheader("종목 통계 (최근 1년)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1년 수익률", f"{stats['return_1y_pct']:+.1f}%")
    c2.metric("연 변동성", f"{stats['volatility_annual_pct']:.1f}%")
    c3.metric("최대 낙폭(MDD)", f"{stats['max_drawdown_1y_pct']:.1f}%")
    c4.metric("월평균 수익률", f"{stats['avg_monthly_return_pct']:+.1f}%")


# ── 메인 흐름 ──────────────────────────────────────────────
result = st.session_state.get("result")

if ticker:
    show_price(ticker)

st.divider()

if generate:
    if not ticker:
        st.error("티커를 입력해줘.")
        st.stop()

    payload = {
        "seed_money": int(seed_money),
        "target_profit": int(target_profit),
        "ticker": ticker,
        "target_period_label": target_period_label,
        "target_period_days": int(target_period_days),
        "days": int(days),
        "max_news": int(max_news),
    }

    try:
        with st.spinner("뉴스 수집 → 심리분석(FinBERT) → 시세 조회 → 시나리오 생성 중..."):
            resp = requests.post(f"{API_URL}/generate", json=payload, timeout=180)
            resp.raise_for_status()
            result = resp.json()
            st.session_state["result"] = result
    except requests.exceptions.ConnectionError:
        st.error("백엔드에 연결 못 함. 다른 터미널에서 `uvicorn main:app --reload` 를 먼저 켜줘.")
        st.stop()
    except Exception as e:
        st.error(f"요청 실패: {e}")
        st.stop()

if result:
    if result.get("news_error"):
        st.warning(f"뉴스 수집 문제: {result['news_error']} — 심리 없이 일반 시나리오로 생성됨.")
    if result.get("quote"):
        show_quote(result["quote"])
    if result.get("stats"):
        show_stats(result["stats"])
    if result.get("sentiment"):
        show_sentiment(result["sentiment"])

    st.subheader("트레이딩 시나리오 · 목표 현실성 분석")
    st.markdown(_clean_scenario(result.get("scenario", "(응답 없음)")))
