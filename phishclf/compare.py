"""Compare four models on the same task over several random splits.

For every seed the data gets a fresh stratified 70/15/15 split. Each model is trained on the
training part, its alert threshold is picked on validation from the same false-positive budget,
and it is scored once on test. Reporting the mean and standard deviation over seeds shows how
much of a difference between models is real and how much is split-to-split noise.

Models:
- ``logreg``: the shipped model (word + character TF-IDF + signals, logistic regression)
- ``linear_svm``: the same features, linear SVM (scores are margins, not probabilities)
- ``naive_bayes``: word TF-IDF, multinomial naive Bayes
- ``fasttext_torch``: hashed word n-grams, learned embeddings, PyTorch (optional)

Probabilistic models also get a Brier score and expected calibration error (ECE): a model whose
"0.9" is right 90% of the time is easier for an analyst to trust than one that is merely ranked
well.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from phishclf.evaluate import metrics_at, pick_threshold, split
from phishclf.model import build_pipeline

REPORTED = ("precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc")


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    """Average gap between predicted probability and observed phishing rate, weighted by bin size."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def _scores(name: str, train, val, test, seed: int, features):
    """Return (validation scores, test scores, is_probability, extra info) for one model."""
    if name == "naive_bayes":
        nb = Pipeline([("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)), ("nb", MultinomialNB())])
        nb.fit(train[0], train[1])
        return nb.predict_proba(val[0])[:, 1], nb.predict_proba(test[0])[:, 1], True, {}
    if name == "fasttext_torch":
        from phishclf.neural import FastTextClassifier

        ft = FastTextClassifier(seed=seed).fit(train[0], train[1], val[0], val[1])
        return ft.predict_proba(val[0])[:, 1], ft.predict_proba(test[0])[:, 1], True, {"epochs": ft.epochs_run}
    xtr, xv, xte = features
    if name == "logreg":
        clf = clone(build_pipeline(seed=seed).named_steps["clf"]).fit(xtr, train[1])
        return clf.predict_proba(xv)[:, 1], clf.predict_proba(xte)[:, 1], True, {}
    if name == "linear_svm":
        svm = LinearSVC(C=0.5, class_weight="balanced", random_state=seed).fit(xtr, train[1])
        return svm.decision_function(xv), svm.decision_function(xte), False, {}
    raise ValueError(f"unknown model {name!r}")


def available_models() -> list[str]:
    models = ["logreg", "linear_svm", "naive_bayes"]
    try:
        import torch  # noqa: F401
    except ImportError:
        return models
    return [*models, "fasttext_torch"]


def compare(texts, labels, seeds=(0, 1, 2, 3, 4), max_fpr: float = 0.01, models=None, log=print) -> dict:
    models = models or available_models()
    runs: dict[str, list[dict]] = {m: [] for m in models}
    for seed in seeds:
        train, val, test = split(texts, labels, seed)
        features = None
        if {"logreg", "linear_svm"} & set(models):
            union = build_pipeline(seed=seed).named_steps["features"]
            features = (union.fit_transform(train[0]), union.transform(val[0]), union.transform(test[0]))
        for name in models:
            start = time.perf_counter()
            val_s, test_s, is_prob, extra = _scores(name, train, val, test, seed, features)
            m = asdict(metrics_at(test[1], test_s, pick_threshold(val[1], val_s, max_fpr)))
            if is_prob:
                m["brier"] = round(float(brier_score_loss(test[1], test_s)), 4)
                m["ece"] = round(expected_calibration_error(test[1], test_s), 4)
            m["seconds"] = round(time.perf_counter() - start, 1)
            m.update(extra)
            runs[name].append(m)
            log(f"seed {seed} {name}: F1 {m['f1']:.3f} FPR {m['false_positive_rate']:.3%} ({m['seconds']}s)")
    summary = {}
    for name, rs in runs.items():
        keys = [k for k in (*REPORTED, "brier", "ece", "seconds") if k in rs[0]]
        summary[name] = {
            k: {
                "mean": round(float(np.mean([r[k] for r in rs])), 4),
                "std": round(float(np.std([r[k] for r in rs])), 4),
            }
            for k in keys
        }
    return {"seeds": list(seeds), "max_fpr_target": max_fpr, "summary": summary, "runs": runs}
