"""Command line: ``phishclf train`` builds and evaluates the model; ``phishclf score`` rates emails.

The model file is a joblib pickle, and loading a pickle can run code. ``train`` therefore writes a
SHA-256 next to the model, and ``score`` refuses a model whose hash does not match. Only load
model files you trained yourself.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import json
import os
import sys

import joblib

from phishclf import data
from phishclf.evaluate import run
from phishclf.model import explain

MAX_EMAIL_BYTES = 5 * 1024 * 1024


def email_text(raw: bytes) -> str:
    """Subject plus plain-text body (HTML parts included as-is) from a raw RFC 822 message or plain text."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    if not msg.keys():  # not an RFC 822 message: treat it as the email body itself
        return raw.decode("utf-8", errors="replace")
    parts = [str(msg.get("subject", ""))]
    for part in msg.walk():
        if part.get_content_maintype() == "text":
            try:
                parts.append(part.get_content())
            except (LookupError, UnicodeError):
                payload = part.get_payload(decode=True) or b""
                parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(p for p in parts if p)


def save_model(pipe, path: str) -> None:
    joblib.dump(pipe, path)
    with open(path + ".sha256", "w") as fh:
        fh.write(data.sha256_of(path) + "\n")


def load_model(path: str):
    try:
        with open(path + ".sha256") as fh:
            expected = fh.read().strip()
    except OSError:
        raise SystemExit(f"phishclf: {path}.sha256 missing; refusing to load an unverified model") from None
    if data.sha256_of(path) != expected:
        raise SystemExit(f"phishclf: {path} does not match its recorded SHA-256; refusing to load it")
    return joblib.load(path)


def cmd_train(args) -> int:
    path = data.download(args.data)
    ds = data.load(path)
    pipe, report = run(ds.texts, ds.labels, max_fpr=args.max_fpr, seed=args.seed)
    report["dataset"] = {
        "raw_rows": ds.raw_rows,
        "dropped_empty": ds.dropped_empty,
        "dropped_duplicates": ds.dropped_duplicates,
        "used": len(ds.texts),
        "sha256": data.DATA_SHA256,
    }
    save_model(pipe, args.model)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    m = report["test_at_chosen_threshold"]
    print(
        f"test: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}, "
        f"FPR {m['false_positive_rate']:.3%}, ROC-AUC {m['roc_auc']:.3f} at threshold {m['threshold']}"
    )
    return 0


def cmd_score(args) -> int:
    pipe = load_model(args.model)
    threshold = args.threshold
    if threshold is None:
        with open(args.report) as fh:
            threshold = json.load(fh)["test_at_chosen_threshold"]["threshold"]
    results = []
    for path in args.emails:
        if os.path.getsize(path) > MAX_EMAIL_BYTES:
            results.append({"file": path, "error": "larger than 5 MB, not scored"})
            continue
        with open(path, "rb") as fh:
            ex = explain(pipe, email_text(fh.read()))
        results.append(
            {
                "file": path,
                "phishing_probability": round(ex.probability, 4),
                "verdict": "phishing" if ex.probability >= threshold else "not phishing",
                "threshold": threshold,
                "top_reasons": [t for t, _ in ex.top_terms],
            }
        )
    print(json.dumps(results if len(results) != 1 else results[0], indent=2))
    return 1 if any(r.get("verdict") == "phishing" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phishclf", description="Explainable phishing-email classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="download the dataset, train, evaluate and save the model")
    t.add_argument("--data", default="data/Phishing_Email.csv")
    t.add_argument("--model", default="model/phishclf.joblib")
    t.add_argument("--report", default="model/report.json")
    t.add_argument("--max-fpr", type=float, default=0.01)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score", help="score .eml or text files; exit 1 if any is phishing")
    s.add_argument("emails", nargs="+")
    s.add_argument("--model", default="model/phishclf.joblib")
    s.add_argument("--report", default="model/report.json")
    s.add_argument("--threshold", type=float)
    args = p.parse_args(argv)
    if args.cmd == "train":
        os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
        return cmd_train(args)
    return cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
