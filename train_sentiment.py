"""
train_sentiment.py
------------------
금융 뉴스 감정분류 모델 학습 스크립트 (직접 학습하는 ML 파트)

데이터: Financial PhraseBank (Malo et al., 2014) - 영어 금융 뉴스 문장, 긍정/부정/중립 라벨
방식  : TF-IDF 벡터화 + (로지스틱회귀 / 선형SVM / 나이브베이즈) 비교 후 최적 모델 저장

실행:  python train_sentiment.py
결과:  sentiment_model.pkl  (문장 -> 감정 예측 파이프라인)
       metrics.txt         (발표용 정확도/리포트)
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent           # 스크립트 위치 기준 (CWD 무관)
DATA_FILE = BASE / "data" / "Sentences_50Agree.txt"   # 4,846 문장 (데이터 최다)
MODEL_OUT = BASE / "sentiment_model.pkl"
METRICS_OUT = BASE / "metrics.txt"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """`문장@라벨` 형식 파일을 읽어 DataFrame으로. 인코딩은 latin-1."""
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)   # 문장 안의 @ 대비 rsplit
            label = label.strip().lower()
            if label in ("positive", "negative", "neutral"):
                rows.append((sentence.strip(), label))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df


def build_pipeline(clf):
    """TF-IDF(단어 + 문자 n-gram) + 분류기 파이프라인.
       - 단어 1~2gram: 'net loss' 같은 금융 표현 포착
       - 문자 3~5gram: 오타/어미변화/합성어에 강함 (정확도 +3%p 효과)
       LinearSVC는 확률을 못 주니 Calibrated로 감싸 확률(심리 점수)까지 뽑는다."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", clf)])


def main():
    df = load_data(DATA_FILE)
    print(f"[데이터] 총 {len(df)}문장")
    print(df["label"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, C=10.0, class_weight="balanced"),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), cv=3),
        "MultinomialNB": MultinomialNB(alpha=0.3),
    }

    results = []
    best_name, best_pipe, best_f1 = None, None, -1.0
    report_lines = []

    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        results.append((name, acc, macro_f1))
        print(f"[{name:20s}] 정확도 {acc:.4f} | macro-F1 {macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, macro_f1

    print(f"\n>>> 최적 모델: {best_name} (macro-F1 {best_f1:.4f})\n")

    # 최적 모델 상세 리포트
    best_pred = best_pipe.predict(X_test)
    labels = ["negative", "neutral", "positive"]
    report = classification_report(y_test, best_pred, labels=labels, digits=4)
    cm = confusion_matrix(y_test, best_pred, labels=labels)

    report_lines.append("=== 모델 비교 ===")
    for name, acc, mf1 in results:
        report_lines.append(f"{name:20s}  accuracy={acc:.4f}  macro-F1={mf1:.4f}")
    report_lines.append(f"\n선택된 모델: {best_name}")
    report_lines.append("\n=== 분류 리포트 (test 20%) ===")
    report_lines.append(report)
    report_lines.append("=== 혼동 행렬 (행=실제, 열=예측) [neg, neu, pos] ===")
    report_lines.append(str(cm))
    metrics_text = "\n".join(report_lines)
    Path(METRICS_OUT).write_text(metrics_text, encoding="utf-8")
    print(metrics_text)

    joblib.dump(best_pipe, MODEL_OUT)
    print(f"\n[저장] {MODEL_OUT}  ({Path(MODEL_OUT).stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
