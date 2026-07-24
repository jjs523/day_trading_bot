#!/usr/bin/env bash
# FastAPI는 컨테이너 "내부 전용"(127.0.0.1)으로 띄운다.
#   0.0.0.0으로 열면 배포처가 8000(백엔드)을 웹 포트로 오해해 "Not Found"가 뜬다.
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Streamlit만 외부로 노출 — 배포처가 주는 $PORT에 바인딩
streamlit run app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0

  