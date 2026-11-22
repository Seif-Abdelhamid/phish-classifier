"""The classifier: TF-IDF text features plus a few hand-written phishing signals, logistic regression.

Logistic regression is chosen on purpose over a bigger model: it trains in seconds on a laptop,
and every score can be explained as the words and signals that pushed it up, which is what an
analyst needs to trust or overrule a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
URGENT_RE = re.compile(
    r"\b(urgent|immediately|verify|suspend(?:ed)?|confirm your|account (?:will be )?(?:locked|closed)|"
    r"password|login|update your|click here|act now|within 24 hours)\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"[$£€]\s?\d|\b(?:wire|transfer|bank|invoice|payment|lottery|winner|prize)\b", re.IGNORECASE)

SIGNAL_NAMES = ("url_count", "ip_url", "urgent_terms", "money_terms", "html", "exclamations", "caps_ratio")


def signals(text: str) -> list[float]:
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    return [
        min(len(URL_RE.findall(text)), 20) / 20,
        1.0 if IP_URL_RE.search(text) else 0.0,
        min(len(URGENT_RE.findall(text)), 10) / 10,
        min(len(MONEY_RE.findall(text)), 10) / 10,
        1.0 if re.search(r"<\s*(html|a\s+href|table|div)\b", text, re.IGNORECASE) else 0.0,
        min(text.count("!"), 10) / 10,
        caps,
    ]


class Signals(BaseEstimator, TransformerMixin):
    """Hand-written phishing signals, scaled to 0..1."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return csr_matrix(np.array([signals(t) for t in x], dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"signal:{n}" for n in SIGNAL_NAMES])


def build_pipeline(c: float = 4.0, seed: int = 0) -> Pipeline:
    words = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=60000, sublinear_tf=True
    )
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000, sublinear_tf=True)
    features = FeatureUnion([("words", words), ("chars", chars), ("signals", Signals())])
    clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Explanation:
    probability: float
    top_terms: list[tuple[str, float]]


def explain(pipe: Pipeline, text: str, k: int = 8) -> Explanation:
    """Phishing probability plus the features that pushed this email toward phishing the most."""
    features: FeatureUnion = pipe.named_steps["features"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    vec = features.transform([text])
    prob = float(clf.predict_proba(vec)[0, 1])
    contrib = vec.multiply(clf.coef_[0]).tocsr()
    names = features.get_feature_names_out()
    idx = contrib.indices[np.argsort(-contrib.data)][:k]
    top = [(str(names[i]).split("__", 1)[-1], float(contrib[0, i])) for i in idx if contrib[0, i] > 0]
    return Explanation(prob, top)
