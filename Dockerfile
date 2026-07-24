FROM python:3.12-slim

WORKDIR /app

# 로그가 즉시 보이도록 (print 버퍼링 끄기)
ENV PYTHONUNBUFFERED=1

# (모델 유형에서 시스템 라이브러리가 필요하면 여기서 설치)
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# PyTorch CPU 버전을 공식 링크에서 가볍게 먼저 설치
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]

