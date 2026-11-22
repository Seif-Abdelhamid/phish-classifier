"""Train, pick an alert threshold, and measure on a held-out test set.

Split: 70% train, 15% validation, 15% test, stratified, after de-duplication. The threshold is
chosen on validation only (the lowest threshold whose false-positive rate is at or below the
target, default 1%), then applied unchanged to the test set. A false positive costs an analyst
time on a real email, so the report leads with precision and false-positive rate, not accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from phishclf.model import build_pipeline


@dataclass
class Metrics:
    """Scores on one split at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    roc_auc: float
    pr_auc: float
    tp: int
    fp: int
    tn: int
    fn: int


def metrics_at(y_true, scores, threshold: float) -> Metrics:
    """Classification metrics when every score >= threshold is called phishing."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return Metrics(
        threshold=round(float(threshold), 4),
        precision=round(float(precision_score(y_true, pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, pred, zero_division=0)), 4),
        false_positive_rate=round(float(fp / (fp + tn)) if fp + tn else 0.0, 4),
        roc_auc=round(float(roc_auc_score(y_true, scores)), 4) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=round(float(average_precision_score(y_true, scores)), 4),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def pick_threshold(y_true, scores, max_fpr: float = 0.01) -> float:
    """Lowest threshold whose false-positive rate on this data is <= max_fpr (highest recall at that FPR)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    negatives = scores[y_true == 0]
    if negatives.size == 0:
        return 0.5
    candidates = np.unique(np.concatenate([scores, [1.0 + 1e-9]]))
    for t in candidates:  # ascending: first one that satisfies the FPR budget maximizes recall
        if float(np.mean(negatives >= t)) <= max_fpr:
            return float(t)
    return 1.0


def split(texts, labels, seed: int = 0):
    """Stratified 70/15/15 train/validation/test split."""
    x_train, x_rest, y_train, y_rest = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def baselines(train, test) -> dict[str, dict]:
    """Majority-class and plain TF-IDF naive Bayes, to show the main model earns its complexity."""
    out = {}
    majority = DummyClassifier(strategy="most_frequent").fit(train[0], train[1])
    out["majority_class"] = asdict(metrics_at(test[1], majority.predict_proba(test[0])[:, 1], 0.5))
    nb = Pipeline([("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)), ("nb", MultinomialNB())])
    nb.fit(train[0], train[1])
    out["tfidf_naive_bayes"] = asdict(metrics_at(test[1], nb.predict_proba(test[0])[:, 1], 0.5))
    return out


def run(texts, labels, max_fpr: float = 0.01, seed: int = 0):
    """Train the main model and return (fitted pipeline, report dict)."""
    train, val, test = split(texts, labels, seed)
    pipe = build_pipeline(seed=seed).fit(train[0], train[1])
    val_scores = pipe.predict_proba(val[0])[:, 1]
    threshold = pick_threshold(val[1], val_scores, max_fpr)
    test_scores = pipe.predict_proba(test[0])[:, 1]
    report = {
        "sizes": {"train": len(train[0]), "validation": len(val[0]), "test": len(test[0])},
        "phishing_share_test": round(float(np.mean(test[1])), 4),
        "max_fpr_target": max_fpr,
        "test_at_chosen_threshold": asdict(metrics_at(test[1], test_scores, threshold)),
        "test_at_0_5": asdict(metrics_at(test[1], test_scores, 0.5)),
        "baselines_test": baselines(train, test),
    }
    return pipe, report
