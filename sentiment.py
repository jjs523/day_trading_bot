"""
sentiment.py
------------
뉴스 심리를 분석하는 추론 모듈. 두 개의 엔진을 지원한다.

  1) FinBERT (ProsusAI/finbert) - 금융 특화 사전학습 BERT. 정확도 높음(기본).
  2) 자체 학습 모델 (sentiment_model.pkl, TF-IDF + LinearSVC) - 우리가 직접 학습.
     FinBERT 사용 불가(미설치/오프라인)일 때 자동 폴백 + 발표용 '비교' 근거.

핵심 함수:
  predict_one(text)            -> 한 문장의 감정 + 확률
  analyze_news(texts)          -> 종합 심리점수(0~100) + 분포 + 개별결과 + 엔진정보
  analyze_news_by_date(items)  -> 날짜별 심리점수 시계열 (가격 차트 오버레이용)

엔진 선택: 환경변수 SENTIMENT_ENGINE = "finbert"(기본) | "sklearn"
"""

import os
from pathlib import Path
from functools import lru_cache

MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pkl"
LABELS = ["negative", "neutral", "positive"]
KO = {"negative": "부정", "neutral": "중립", "positive": "긍정"}

ENGINE = os.getenv("SENTIMENT_ENGINE", "finbert").lower()

# 엔진별 발표용 정확도 라벨 (프롬프트/UI 표시에 사용)
ENGINE_LABELS = {
    "finbert": ("FinBERT (금융 특화 BERT)", "약 88~90%"),
    "sklearn": ("자체 학습 모델 (TF-IDF+LinearSVC)", "약 77.9%"),
}

FINBERT_MODEL = "ProsusAI/finbert"


# ── 엔진 1: FinBERT ────────────────────────────────────────
@lru_cache(maxsize=1)
def _finbert():
    """FinBERT 파이프라인을 한 번만 로드. 실패하면 None."""
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TextClassificationPipeline)
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    return TextClassificationPipeline(
        model=mdl, tokenizer=tok, top_k=None, truncation=True, function_to_apply="softmax")


def _predict_finbert(text: str) -> dict:
    pipe = _finbert()
    scores = pipe(text)[0]                       # [{'label':'positive','score':..}, ...]
    probs = {s["label"].lower(): round(float(s["score"]), 4) for s in scores}
    for c in LABELS:                              # 누락 방지
        probs.setdefault(c, 0.0)
    label = max(probs, key=probs.get)
    return _pack(text, label, probs)


# ── 엔진 2: 자체 학습 sklearn 모델 ──────────────────────────
@lru_cache(maxsize=1)
def _sklearn():
    import joblib
    return joblib.load(MODEL_PATH)


def _predict_sklearn(text: str) -> dict:
    m = _sklearn()
    proba = m.predict_proba([text])[0]
    probs = {cls: round(float(p), 4) for cls, p in zip(m.classes_, proba)}
    for c in LABELS:
        probs.setdefault(c, 0.0)
    label = m.predict([text])[0]
    return _pack(text, label, probs)


# ── 공통 ───────────────────────────────────────────────────
def _pack(text: str, label: str, probs: dict) -> dict:
    return {
        "text": text,
        "label": label,
        "label_ko": KO[label],
        "confidence": round(float(max(probs.values())), 4),
        "probs": probs,
    }


@lru_cache(maxsize=1)
def _active_engine() -> str:
    """실제로 쓸 엔진 결정. finbert 선택인데 사용 불가하면 sklearn 으로 자동 전환."""
    if ENGINE == "finbert":
        try:
            _finbert()
            return "finbert"
        except ImportError:
            # 배포 등 transformers/torch 미설치 환경 → 의도된 자체 모델 사용 (에러 아님)
            print("[정보] transformers 미설치 환경 → 자체 학습 sklearn 모델 사용")
            return "sklearn"
        except Exception as e:
            print(f"[알림] FinBERT 사용 불가 → 자체 sklearn 모델 사용 ({e})")
            return "sklearn"
    return "sklearn"


def predict_one(text: str) -> dict:
    return _predict_finbert(text) if _active_engine() == "finbert" else _predict_sklearn(text)


def _score_from_items(items: list[dict]) -> float:
    """각 뉴스의 (P(긍정)-P(부정)) 평균을 0~100 심리점수로 변환. 50=중립."""
    net = sum(it["probs"].get("positive", 0) - it["probs"].get("negative", 0)
              for it in items) / len(items)
    return max(0.0, min(100.0, round(50 + 50 * net, 1)))


def _verdict(score: float) -> str:
    if score >= 60:
        return "긍정 우세"
    if score <= 40:
        return "부정 우세"
    return "중립/혼조"


def analyze_news(texts: list[str]) -> dict:
    """뉴스 여러 건을 분석해 종합 심리를 계산."""
    engine = _active_engine()
    name, acc = ENGINE_LABELS[engine]

    if not texts:
        return {"sentiment_score": 50.0, "verdict": "중립",
                "counts": {"부정": 0, "중립": 0, "긍정": 0}, "n": 0, "items": [],
                "engine": engine, "engine_name": name, "engine_accuracy": acc}

    items = [predict_one(t) for t in texts]
    score = _score_from_items(items)
    counts = {KO[c]: sum(1 for it in items if it["label"] == c) for c in LABELS}

    return {
        "sentiment_score": score,
        "verdict": _verdict(score),
        "counts": counts,
        "n": len(items),
        "items": items,
        "engine": engine,
        "engine_name": name,          # 예: "FinBERT (금융 특화 BERT)"
        "engine_accuracy": acc,       # 예: "약 88~90%"
    }


def analyze_news_by_date(articles: list[dict]) -> list[dict]:
    """
    날짜별 심리점수 시계열 (가격 차트 위 심리 오버레이용).
    articles: [{"text":..., "publishedAt": "2026-07-22T...Z"}, ...]
    반환: [{"date": "2026-07-22", "score": 61.2, "n": 4}, ...]  날짜 오름차순
    """
    buckets: dict[str, list[dict]] = {}
    for a in articles:
        ts = a.get("publishedAt") or ""
        day = ts[:10]                            # "YYYY-MM-DD"
        if not day:
            continue
        buckets.setdefault(day, []).append(predict_one(a["text"]))

    out = []
    for day in sorted(buckets):
        items = buckets[day]
        out.append({"date": day, "score": _score_from_items(items), "n": len(items)})
    return out


if __name__ == "__main__":
    sample = [
        "Apple beats quarterly earnings estimates as iPhone sales surge.",
        "Tesla shares tumble after the company warns of weaker demand ahead.",
        "The company announced a stock buyback program worth $10 billion.",
        "Regulators launched an investigation into the firm's accounting practices.",
        "The board approved the annual budget for the next fiscal year.",
    ]
    result = analyze_news(sample)
    print(f"엔진: {result['engine_name']} (정확도 {result['engine_accuracy']})")
    print(f"종합 심리 점수: {result['sentiment_score']} / 100  ({result['verdict']})")
    print(f"분포: {result['counts']}\n")
    for it in result["items"]:
        print(f"  [{it['label_ko']} {it['confidence']*100:.0f}%] {it['text']}")
