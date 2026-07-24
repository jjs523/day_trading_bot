FROM python:3.12-slim

WORKDIR /app

# 로그가 즉시 보이도록 (print 버퍼링 끄기)
ENV PYTHONUNBUFFERED=1

# 배포는 512MB 무료 인스턴스에 맞춰 '가벼운 자체 학습 sklearn 모델'만 사용한다.
# (FinBERT/torch/transformers는 RAM 초과로 OOM → 배포판에서 제외)
ENV SENTIMENT_ENGINE=sklearn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 배포 환경의 sklearn 버전에 맞춰 감정분류 모델을 새로 학습(버전 불일치 방지, 약 30초).
# 실패하면 저장소에 커밋된 sentiment_model.pkl 을 그대로 사용.
RUN python train_sentiment.py || echo "학습 실패 - 커밋된 pkl 사용"

RUN chmod +x start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]
