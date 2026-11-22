# phish-classifier

An explainable phishing-email classifier built with scikit-learn. It scores an email (a raw
`.eml` file or plain text), says **why** (the words and signals that pushed the score up), and
alerts at a threshold chosen from a **false-positive budget** instead of an arbitrary 0.5.

```
$ phishclf score examples/phish.eml
{
  "phishing_probability": 0.9987,
  "verdict": "phishing",
  "threshold": 0.463,
  "top_reasons": ["signal:urgent_terms", "signal:ip_url", "your", "click here", "signal:exclamations", ...]
}
```

## The problem

A university or company security team receives a steady stream of user-reported suspicious
emails. Each one costs an analyst a few minutes. A model can sort that queue, but only if:

1. its **false-positive rate is controlled**, because every false alarm is analyst time spent on a
   real, harmless email;
2. it **explains each verdict**, so an analyst can trust or overrule it in seconds; and
3. its reported accuracy is **honest**. Public phishing datasets contain many duplicate emails.
   If the same email is in both the training and the test set, the test score measures memory,
   not detection.

**Intended users:** SOC analysts and students who triage reported phishing, and anyone who wants a
small, auditable baseline before adopting a commercial filter.

## Existing solutions and the gap

| Option                                                                       | What it does well                                | Gap this project addresses                                                                   |
| ---------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Commercial email security (e.g. Microsoft Defender for Office 365, Abnormal) | Large-scale, behavioral signals, inline blocking | Closed models; you cannot see or tune how a verdict was reached                              |
| SpamAssassin                                                                 | Mature rules plus a Bayes filter                 | Built for spam rather than phishing; no threshold policy by false-positive budget            |
| Typical notebook classifiers on public datasets                              | Quick to try                                     | Usually report accuracy at 0.5 on data with duplicates across splits, which inflates results |

This project is a teaching-grade, reproducible baseline, not a replacement for a mail gateway.

## How it works

```
.eml / text ─► subject + text parts ─► features ─────────────────────────► logistic regression ─► probability
                                        ├ word TF-IDF (1-2 grams)                  │
                                        ├ character TF-IDF (3-5 grams, catches     ├─► threshold from FPR budget ─► verdict
                                        │  obfuscation such as "p@ssw0rd")         │
                                        └ 7 hand-written signals: URL count,       └─► top contributing features ─► reasons
                                          IP-address URLs, urgency terms,
                                          money terms, HTML, "!", capital letters
```

- **Logistic regression on purpose.** It trains in about a minute on a laptop, and each score
  decomposes into per-feature contributions, so the reasons are exact rather than estimated.
- **De-duplication before splitting.** Emails are normalized (whitespace, case) and hashed; exact
  duplicates and empty rows are removed before any split.
- **Threshold from a false-positive budget.** Data is split 70/15/15 (train/validation/test,
  stratified). The threshold is the lowest score whose false-positive rate on _validation_ is at
  most the budget (default 1%). It is then applied unchanged to the _test_ set.
- **Safe loading.** The dataset is fetched over HTTPS only and must match a pinned SHA-256. The
  trained model is a joblib pickle (which can execute code when loaded), so `train` records its
  SHA-256 and `score` refuses a model that does not match. Only load models you trained.

## Results

Dataset: [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
(LGPL-3.0; 18,650 rows, mirrored from a Kaggle collection). Not redistributed here;
`phishclf train` downloads and verifies it.

- 18,650 rows → 552 empty and 585 duplicate rows removed → **17,513 emails** (12,259 train /
  2,627 validation / 2,627 test).
- Test set, at the threshold picked on validation for a 1% false-positive budget (0.463):

| Model                                                                | Precision | Recall    | F1        | False-positive rate | ROC-AUC   |
| -------------------------------------------------------------------- | --------- | --------- | --------- | ------------------- | --------- |
| **This model** (TF-IDF words + chars + signals, logistic regression) | **0.974** | **0.995** | **0.984** | 1.58%               | **0.999** |
| Baseline: TF-IDF + naive Bayes (threshold 0.5)                       | 0.999     | 0.815     | 0.897     | 0.06%               | 0.997     |
| Baseline: always "not phishing"                                      | 0         | 0         | 0         | 0%                  | 0.5       |

Confusion matrix on the 2,627 test emails: 976 phishing caught, 5 missed, 26 false alarms, 1,620
correctly passed.

**Read these numbers carefully.**

- The 1% false-positive budget was met on validation but **not on test (1.58%)**: a threshold
  tuned on 2,627 emails moves when applied to new ones. In production, re-check the false-positive
  rate on your own recent mail and re-pick the threshold.
- Naive Bayes is more precise but misses about 1 in 5 phishing emails. This model trades a few more
  false alarms for catching almost all of them, which suits a triage queue that an analyst reviews.
- A ROC-AUC this high says this dataset is easy to separate, not that phishing is solved. It is one
  public collection, and its mail may look very different from your organization's. Expect lower
  scores on real reported mail.

## Setup and usage

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
phishclf train                                  # downloads + verifies the dataset, trains, writes model/ and model/report.json
phishclf score examples/phish.eml               # JSON verdict with reasons; exit code 1 if phishing
phishclf score suspicious/*.eml --threshold 0.8 # stricter, fewer alerts
```

`phishclf train --max-fpr 0.005` picks a stricter threshold. The full metrics, including both
baselines and the threshold used, are written to `model/report.json`.

## Development

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest --cov=phishclf # 13 tests on synthetic emails (no download needed)
ruff check . && ruff format --check .
bandit -c pyproject.toml -r phishclf
```

CI runs the tests on Python 3.11 to 3.13, Ruff, Bandit and pip-audit.

## Limitations

- English text only; images, attachments and sender reputation are not used.
- Headers other than the subject are ignored, so SPF/DKIM/DMARC failures are not used as signals.
- One public dataset; not validated on any organization's real mail.
- A static model: attackers adapt, so retrain on recent reported emails.

## Adopting or extending it

- **Your own data:** export reported emails labeled by your analysts to a CSV with `Email Text`
  and `Email Type` (`Phishing Email` / `Safe Email`) columns, and point `--data` at it (update the
  pinned hash in `phishclf/data.py`, or pass your own to `download`).
- **More signals:** add to `signals()` in `phishclf/model.py`, e.g. header checks (SPF/DKIM
  results, display-name and domain mismatch) or URL features (newly registered domains).
- **In a SOAR workflow:** `phishclf score` prints JSON and returns exit code 1 for phishing, so a
  playbook can call it and route the email to a queue by verdict.

## License

MIT. The dataset is not included and keeps its own license (LGPL-3.0).
