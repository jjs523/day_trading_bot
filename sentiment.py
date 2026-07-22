"""
sentiment.py
------------
학습된 감정분류 모델을 불러와 뉴스 심리를 분석하는 추론 모듈.
FastAPI / 트레이딩 봇에서 import 해서 쓰는 부분.

핵심 함수:
  predict_one(text)      -> 한 문장의 감정 + 확률
  analyze_news(texts)    -> 뉴스 리스트의 종합 심리 점수(0~100) + 분포 + 개별결과
"""

import joblib
from pathlib import Path
from functools import lru_cache

MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pkl"
LABELS = ["negative", "neutral", "positive"]
KO = {"negative": "부정", "neutral": "중립", "positive": "긍정"}


@lru_cache(maxsize=1)
def _model():
    """모델은 한 번만 로드 (서버에서 매 요청마다 다시 안 읽게)."""
    return joblib.load(MODEL_PATH)


def predict_one(text: str) -> dict:
    m = _model()
    label = m.predict([text])[0]
    proba = m.predict_proba([text])[0]
    probs = {cls: round(float(p), 4) for cls, p in zip(m.classes_, proba)}
    return {
        "text": text,
        "label": label,
        "label_ko": KO[label],
        "confidence": round(float(max(proba)), 4),
        "probs": probs,
    }


def analyze_news(texts: list[str]) -> dict:
    """
    뉴스 여러 건을 분석해 종합 심리를 계산.

    심리 점수(sentiment_score): 0~100
      각 뉴스의 (P(긍정) - P(부정)) 평균을 0~100으로 변환.
      50 = 중립, >50 긍정 우세, <50 부정 우세.
    """
    if not texts:
        return {"sentiment_score": 50.0, "verdict": "중립",
                "counts": {"부정": 0, "중립": 0, "긍정": 0}, "n": 0, "items": []}

    items = [predict_one(t) for t in texts]

    # 확률 기반 종합 점수 (라벨 카운트보다 부드럽고 정보량 많음)
    net = sum(it["probs"].get("positive", 0) - it["probs"].get("negative", 0)
              for it in items) / len(items)
    score = round(50 + 50 * net, 1)          # 0~100
    score = max(0.0, min(100.0, score))

    counts = {KO[c]: sum(1 for it in items if it["label"] == c) for c in LABELS}

    if score >= 60:
        verdict = "긍정 우세"
    elif score <= 40:
        verdict = "부정 우세"
    else:
        verdict = "중립/혼조"

    return {
        "sentiment_score": score,   # LLM 프롬프트에 이 숫자를 넣으면 됨
        "verdict": verdict,
        "counts": counts,           # {'부정': n, '중립': n, '긍정': n}
        "n": len(items),
        "items": items,             # 개별 뉴스 결과 (UI 표/그래프용)
    }


if __name__ == "__main__":
    # 실제 미국주식 뉴스 스타일 샘플로 테스트
    sample = [
        "Apple beats quarterly earnings estimates as iPhone sales surge.",
        "Tesla shares tumble after the company warns of weaker demand ahead.",
        "The company announced a stock buyback program worth $10 billion.",
        "Regulators launched an investigation into the firm's accounting practices.",
        "The board approved the annual budget for the next fiscal year.",
    ]
    result = analyze_news(sample)
    print(f"종합 심리 점수: {result['sentiment_score']} / 100  ({result['verdict']})")
    print(f"분포: {result['counts']}\n")
    for it in result["items"]:
        print(f"  [{it['label_ko']} {it['confidence']*100:.0f}%] {it['text']}")
