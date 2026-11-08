# Gemini evaluation

2026-11-08 · `gemini-3.5-flash-lite` (free tier)

## Receipt extraction — CORD v2 test set

Real receipt photos with hand-labelled fields (Indonesian rupiah), never used for training.

| Check | Result |
|---|---|
| Local classifier says "receipt" | 100/100 (100%) |
| Total amount exactly right | 92/95 (97%) |
| Number of line items exactly right | 83/100 (83%) |
| Number of line items within ±1 | 93/100 (93%) |
| API errors | 0 |

Median time per receipt (classify + extract): 2.2 s

## Trained classifier vs. asking Gemini

The same 70 test pages (10 per type), classified by the local model and by Gemini with no examples (zero-shot).

| | Trained classifier | Gemini zero-shot |
|---|---|---|
| Accuracy | 62/70 (89%) | 59/70 (84%) |
| Median time per page | 37 ms (CPU) | 1380 ms (API) |
| Cost | free, runs offline | API call per page |

| Type | Classifier | Gemini |
|---|---|---|
| Email | 9/10 | 8/10 |
| Form | 9/10 | 9/10 |
| Invoice | 8/10 | 8/10 |
| Letter | 9/10 | 10/10 |
| Receipt | 10/10 | 10/10 |
| Resume / CV | 8/10 | 5/10 |
| Scientific paper | 9/10 | 9/10 |

### What went wrong

- 2 receipts printed totals like `40.000` (Indonesian thousands separator) and the model returned 40. The extraction prompt now explains dot-thousands formats; this fix has **not** been re-measured on this test set (re-tuning on the test set would inflate the score).
- 1 receipt: the model returned the rounded cash amount (62,000) instead of the exact total (61,799).
- Earlier runs scored 88/95 because the *scoring script* misread `365.000,00` as 36,500,000 — fixed in `amount()` and re-scored from the saved answers (`--rescore`, no new API calls).
