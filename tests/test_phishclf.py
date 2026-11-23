import csv
import json
from pathlib import Path

import numpy as np
import pytest

from phishclf import cli, data
from phishclf.evaluate import metrics_at, pick_threshold, run, split
from phishclf.model import build_pipeline, explain, signals

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

PHISH = [
    "URGENT verify your password now at http://10.0.0.{i}/login or your account will be suspended",
    "Click here to confirm your login within 24 hours http://198.51.100.{i}/secure act now!",
    "You are a lottery winner! Wire a transfer fee of ${i}00 to claim your prize immediately",
    "Your bank account is locked. Update your payment details here http://203.0.113.{i}/bank",
]
SAFE = [
    "Meeting notes from Tuesday {i}: the team agreed to ship the report next week",
    "Hi, attached is the draft agenda for the seminar on day {i}, comments welcome",
    "Reminder: the library closes early on Friday {i} for maintenance",
    "Thanks for the review on chapter {i}, I pushed the fixes to the shared folder",
]


def corpus(n=30):
    texts, labels = [], []
    for i in range(n):
        for t in PHISH:
            texts.append(t.format(i=i))
            labels.append(1)
        for t in SAFE:
            texts.append(t.format(i=i))
            labels.append(0)
    return texts, labels


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "Email Text", "Email Type"])
        for i, (text, label) in enumerate(rows):
            w.writerow([i, text, label])


# -- data ---------------------------------------------------------------------------------------
def test_load_drops_empty_and_duplicates_before_any_split(tmp_path):
    p = tmp_path / "d.csv"
    write_csv(
        p,
        [
            ("Verify your password", "Phishing Email"),
            ("  verify   YOUR password ", "Phishing Email"),  # duplicate after normalizing
            ("", "Safe Email"),
            ("empty", "Safe Email"),
            ("Lunch at noon?", "Safe Email"),
        ],
    )
    ds = data.load(str(p))
    assert ds.labels == [1, 0]
    assert (ds.raw_rows, ds.dropped_empty, ds.dropped_duplicates) == (5, 2, 1)


def test_load_rejects_unknown_labels_and_wrong_columns(tmp_path):
    p = tmp_path / "d.csv"
    write_csv(p, [("hello", "Spam")])
    with pytest.raises(data.DataError, match="unknown label"):
        data.load(str(p))
    p.write_text("a,b\n1,2\n")
    with pytest.raises(data.DataError, match="expected columns"):
        data.load(str(p))


def test_download_refuses_a_file_with_the_wrong_hash(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("tampered")
    with pytest.raises(data.DataError, match="does not match"):
        data.download(str(p))


def test_download_refuses_plain_http(tmp_path):
    with pytest.raises(data.DataError, match="non-HTTPS"):
        data.download(str(tmp_path / "x.csv"), url="http://example.com/x.csv")


# -- features and model -------------------------------------------------------------------------
def test_signals_fire_on_phishing_markers():
    s = dict(zip(("urls", "ip", "urgent", "money", "html", "bang", "caps"), signals(PHISH[0].format(i=1)), strict=True))
    assert s["ip"] == 1.0 and s["urls"] > 0 and s["urgent"] > 0
    quiet = signals("see you at lunch")
    assert quiet[:6] == [0.0] * 6


def test_model_learns_and_explains(tmp_path):
    texts, labels = corpus()
    pipe = build_pipeline().fit(texts, labels)
    ex = explain(pipe, "Urgent: verify your password at http://10.9.9.9/login now")
    assert ex.probability > 0.5
    assert ex.top_terms and all(w > 0 for _, w in ex.top_terms)
    assert explain(pipe, "Notes from the seminar meeting, comments welcome").probability < 0.5


# -- evaluation ---------------------------------------------------------------------------------
def test_pick_threshold_respects_the_fpr_budget():
    y = np.array([0] * 100 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.9, 100), np.full(10, 0.95)])
    t = pick_threshold(y, scores, max_fpr=0.01)
    assert np.mean(scores[y == 0] >= t) <= 0.01
    assert np.all(scores[y == 1] >= t)  # still catches every positive


def test_metrics_counts():
    m = metrics_at([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.false_positive_rate == 0.5


def test_split_is_stratified_and_disjoint():
    texts, labels = corpus()
    (xtr, _ytr), (xv, _yv), (xte, yte) = split(texts, labels)
    assert not (set(xtr) & set(xte)) and not (set(xv) & set(xte))
    assert abs(np.mean(yte) - np.mean(labels)) < 0.05


def test_run_reports_test_metrics_and_baselines():
    texts, labels = corpus()
    _, report = run(texts, labels)
    assert set(report["baselines_test"]) == {"majority_class", "tfidf_naive_bayes"}
    assert report["test_at_chosen_threshold"]["recall"] > 0.9


# -- CLI ----------------------------------------------------------------------------------------
def test_email_text_reads_subject_and_body():
    raw = (EXAMPLES / "phish.eml").read_bytes()
    text = cli.email_text(raw)
    assert "URGENT" in text and "192.0.2.15" in text
    assert cli.email_text(b"just a body") == "just a body"


def test_score_refuses_a_tampered_model(tmp_path):
    texts, labels = corpus()
    model = tmp_path / "m.joblib"
    cli.save_model(build_pipeline().fit(texts, labels), str(model))
    with open(model, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit, match="does not match"):
        cli.load_model(str(model))


def test_score_end_to_end(tmp_path, capsys):
    texts, labels = corpus()
    model = tmp_path / "m.joblib"
    cli.save_model(build_pipeline().fit(texts, labels), str(model))
    code = cli.main(["score", str(EXAMPLES / "phish.eml"), "--model", str(model), "--threshold", "0.5"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["verdict"] == "phishing" and out["top_reasons"]
    code = cli.main(["score", str(EXAMPLES / "benign.eml"), "--model", str(model), "--threshold", "0.5"])
    assert code == 0


# -- model comparison ---------------------------------------------------------------------------


def test_expected_calibration_error():
    from phishclf.compare import expected_calibration_error

    assert expected_calibration_error([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert expected_calibration_error([0, 0, 0, 0], [0.9, 0.9, 0.9, 0.9]) == pytest.approx(0.9)


def test_compare_sklearn_models_over_seeds():
    from phishclf.compare import compare

    texts, labels = corpus()
    report = compare(texts, labels, seeds=(0, 1), models=["logreg", "linear_svm", "naive_bayes"], log=lambda m: None)
    assert set(report["summary"]) == {"logreg", "linear_svm", "naive_bayes"}
    assert len(report["runs"]["logreg"]) == 2
    assert report["summary"]["logreg"]["f1"]["mean"] > 0.9
    assert "brier" in report["summary"]["logreg"] and "brier" not in report["summary"]["linear_svm"]


def test_fasttext_torch_learns():
    pytest.importorskip("torch")
    from phishclf.neural import FastTextClassifier, token_ids

    assert token_ids("Click here!", 1000) == token_ids("click HERE !", 1000)  # stable hashing
    texts, labels = corpus()
    ft = FastTextClassifier(buckets=1 << 12, dim=16, epochs=15, seed=0).fit(texts, labels, texts, labels)
    prob = ft.predict_proba(texts)[:, 1]
    assert np.mean((prob >= 0.5) == np.array(labels)) > 0.95
