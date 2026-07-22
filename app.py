"""
app.py - Streamlit UI
뉴스 심리 기반 트레이딩 시나리오 봇의 프런트엔드.
FastAPI(main.py)를 먼저 켜둔 상태에서 실행:
  터미널 1:  uvicorn main:app --reload
  터미널 2:  streamlit run app.py
"""
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="뉴스 심리 트레이딩 봇", page_icon="📈", layout="centered")
st.title("📈 뉴스 심리 기반 트레이딩 시나리오")
st.caption("직접 학습한 감정분류 ML(정확도 77.9%)이 종목 뉴스 심리를 분석하고, "
           "그 결과를 GPT-5 mini가 반영해 시나리오를 생성합니다.")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("관심 티커", value="TSLA").strip().upper()
    seed_money = st.number_input("시드머니 (원)", min_value=0, value=1_000_000, step=100_000)
with col2:
    trade_type = st.selectbox("매매 방식", ["데이 트레이딩", "오버나잇"])
    target_profit = st.number_input("일일 목표수익 (원)", min_value=0, value=30_000, step=10_000)

with st.expander("고급 설정"):
    days = st.slider("뉴스 조회 기간 (일)", 1, 30, 7)
    max_news = st.slider("분석할 뉴스 수", 5, 50, 20)

go = st.button("🚀 시나리오 생성", type="primary", use_container_width=True)


def show_sentiment(senti: dict):
    score = senti["sentiment_score"]
    st.subheader("뉴스 심리 분석")
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


if go:
    if not ticker:
        st.error("티커를 입력해줘.")
        st.stop()

    payload = {
        "seed_money": int(seed_money),
        "target_profit": int(target_profit),
        "ticker": ticker,
        "trade_type": trade_type,
        "days": int(days),
        "max_news": int(max_news),
    }

    try:
        with st.spinner("뉴스 수집 → 심리분석 → 시나리오 생성 중..."):
            resp = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("백엔드에 연결 못 함. 다른 터미널에서 `uvicorn main:app --reload` 를 먼저 켜줘.")
        st.stop()
    except Exception as e:
        st.error(f"요청 실패: {e}")
        st.stop()

    if data.get("news_error"):
        st.warning(f"뉴스 수집 문제: {data['news_error']} — 심리 없이 일반 시나리오로 생성됨.")

    if data.get("sentiment"):
        show_sentiment(data["sentiment"])

    st.subheader("트레이딩 시나리오")
    st.markdown(data.get("scenario", "(응답 없음)"))