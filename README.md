## 📥 Dataset

**[Online Retail II — download here (Kaggle mirror)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)**

Original source / citation: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed CC BY 4.0.

Place the downloaded file at `data/raw/online_retail_II.csv`.

---

# 🔍 Model Stress-Test & Failure Catalog

Most portfolios stop at reporting a model's accuracy or recall. This one doesn't trust that number — it deliberately goes looking for where the [order-cancellation classifier from a previous project](https://github.com/sidv11/eda-deep-dive-online-retail) breaks, and documents exactly why.

## The headline finding

The model has quietly learned **"small order = risky, big order = safe"** — and it's not a soft, borderline pattern. It's confident and specific:

| | Missed cancellations | Caught cancellations |
|---|---|---|
| Median items in order | 4 | 1 |
| Median order value | £113.55 | £14.93 |
| Model's median confidence | ~20% *(should be near 100%)* | — |

A hand-crafted **£540, 18-item order gets a 2.8% cancellation probability** — the model would be badly, confidently wrong if that order were actually cancelled. A tiny £8.50 single-item order gets 92.6% — correctly flagged. The model isn't equally trustworthy everywhere; it's reliable on simple orders and blind on large, varied ones — exactly the orders where a missed cancellation costs the business the most.

## What this project actually does

1. **Rebuilds the exact Day 8 model** (same features, same LightGBM + SMOTE pipeline) — nothing new to stress-test, the point is finding real weaknesses in existing work
2. **Pulls out real false negatives** from the test set and statistically compares them against what the model *did* catch — the blind spot above is measured, not guessed
3. **Checks confidence, not just correctness** — distinguishes "narrowly unsure and wrong" from "confidently, badly wrong" (it's the latter)
4. **Builds hand-crafted adversarial cases** to confirm the pattern in numbers anyone can sanity-check
5. **Tests a real, quantified mitigation** — lowering the decision threshold — and reports the actual trade-off (more recall, more false alarms) instead of claiming a free fix

## Full failure catalog

| Case | Model says | Confidence | Verdict |
|---|---|---|---|
| Tiny single-item order (£8.50) | Cancelled | 92.6% | ✅ Correct |
| **Large bulk order (£540, 18 items)** | Not Cancelled | **2.8%** | 🚨 The blind spot |
| Medium multi-item order (£120, 5 items) | Not Cancelled | 6.6% | ⚠️ Same blind spot, milder |
| Rare-country small order (£15) | Cancelled | 82.0% | ✅ Order size dominates correctly |
| **Weekend large order (£610, 20 items)** | Not Cancelled | **3.5%** | 🚨 Timing doesn't rescue it |
| Zero-product edge case (invalid input) | Not Cancelled | 0.1% | ✅ Degrades sensibly, doesn't crash |

## The threshold trade-off, quantified

| Decision threshold | Recall | False alarm rate |
|---|---|---|
| 0.50 (default) | 86.6% | 7.9% |
| 0.35 | 90.0% | 9.5% |
| 0.25 | 92.2% | 11.1% |
| 0.15 | 94.3% | 13.5% |

Lowering the threshold genuinely helps recall — but it's a real trade-off the business has to decide on, not a free improvement.

## What's in this repo

```
model-failure-catalog/
├── data/
│   └── raw/
│       └── online_retail_II.csv
├── features.py        ← reused from the classification project (order-level feature engineering)
├── model_utils.py       ← rebuilds the exact Day 8 model so it can be stress-tested consistently
├── notebook.ipynb         ← the full stress-test: real failures → pattern → synthetic confirmation → mitigation
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
```
Then open `notebook.ipynb` and run it top to bottom. Every cell was verified end-to-end before this was pushed.

## What I'd do next

The real fix isn't threshold-tuning alone — it's adding a feature the model currently can't see: a customer's own order-size history, so a "large order" is judged as large *for that customer*, not in absolute terms. That's a bigger change than this project's scope, but it's the direction a genuine fix would take.
