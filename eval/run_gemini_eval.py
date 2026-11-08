"""Evaluate the parts that use Gemini (needs GEMINI_API_KEY; throttled for the free tier).

1. Receipt extraction on the CORD v2 test set (100 labelled receipt photos): is the total right?
   Are the line items found? Does the local classifier call them receipts?
2. Classifier vs. zero-shot Gemini on the same test pages: accuracy and speed.

    python -m eval.run_gemini_eval                 # ~15 minutes
    python -m eval.run_gemini_eval --receipts 10 --per-class 2
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from docsort.classes import CLASSES, LABELS
from docsort.config import get_settings
from docsort.data_prep import CORD_REPO, CORD_SPLITS
from docsort.extract import ZERO_SHOT_SYSTEM, ZeroShotType
from docsort.gemini import GeminiError, build_gemini
from docsort.service import DocSorter, load_page

HERE = Path(__file__).resolve().parent


def amount(text) -> int | None:
    """CORD prices are Indonesian rupiah written like '60,000', '60.000' or '365.000,00' → whole rupiah."""
    if text is None:
        return None
    text = re.sub(r"[.,]\d{2}\s*$", "", str(text).strip())  # drop a 2-digit cents part (",00")
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def cord_truth(raw: str) -> dict:
    gt = json.loads(raw)["gt_parse"]
    menu = gt.get("menu") or []
    menu = [menu] if isinstance(menu, dict) else menu
    total = (gt.get("total") or {})
    total = total[0] if isinstance(total, list) else total
    return {"total": amount(total.get("total_price")), "items": len(menu)}


def eval_receipts(sorter: DocSorter, n: int, pause: float) -> dict:
    s = get_settings()
    path = hf_hub_download(CORD_REPO, CORD_SPLITS["test"][0], repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
    table = pq.read_table(path)
    rows = list(zip(table.column("image").to_pylist(), table.column("ground_truth").to_pylist()))[:n]
    out = []
    for i, (img, gt_raw) in enumerate(rows, 1):
        truth = cord_truth(gt_raw)
        data = img["bytes"] if isinstance(img, dict) else img
        r = sorter.process(data, f"cord-test-{i}.jpg")
        f = r.fields or {}
        pred_total = round(f["total"]) if isinstance(f.get("total"), (int, float)) else None
        row = {"id": i, "classified_as": r.classification.doc_type, "confidence": r.classification.confidence,
               "true_total": truth["total"], "pred_total": pred_total, "true_items": truth["items"],
               "pred_items": len(f.get("items") or []), "error": r.extraction_error, "latency_ms": r.latency_ms}
        out.append(row)
        print(f"receipt {i}/{len(rows)} type={row['classified_as']} total {pred_total} vs {truth['total']}", flush=True)
        time.sleep(pause)
    return summarize(out)


def rescore(saved: dict) -> dict:
    """Recompute the ground truth for saved answers (e.g. after fixing amount())."""
    s = get_settings()
    path = hf_hub_download(CORD_REPO, CORD_SPLITS["test"][0], repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
    truths = [cord_truth(raw) for raw in pq.read_table(path, columns=["ground_truth"]).column("ground_truth").to_pylist()]
    rows = saved["rows"]
    for r in rows:
        t = truths[r["id"] - 1]
        r["true_total"], r["true_items"] = t["total"], t["items"]
    return summarize(rows)


def summarize(out: list[dict]) -> dict:
    with_total = [r for r in out if r["true_total"] is not None]
    return {
        "n": len(out),
        "classified_as_receipt": sum(r["classified_as"] == "receipt" for r in out),
        "with_true_total": len(with_total),
        "total_exact": sum(r["pred_total"] == r["true_total"] for r in with_total),
        "items_exact": sum(r["pred_items"] == r["true_items"] for r in out),
        "items_within_1": sum(abs(r["pred_items"] - r["true_items"]) <= 1 for r in out),
        "errors": sum(r["error"] is not None for r in out),
        "median_latency_ms": statistics.median(r["latency_ms"] for r in out),
        "rows": out,
    }


def eval_zero_shot(sorter: DocSorter, per_class: int, pause: float) -> dict:
    gem = sorter.gemini
    random.seed(11)
    root = get_settings().data_dir / "images" / "test"
    sample = [(c, p) for c in CLASSES for p in random.sample(sorted((root / c).glob("*.jpg")), per_class)]
    rows = []
    for i, (truth, path) in enumerate(sample, 1):
        data = path.read_bytes()
        page, image, mime = load_page(data)
        t0 = time.perf_counter()
        local = sorter.classify(page).doc_type
        t1 = time.perf_counter()
        try:
            remote = gem.read_image(image=image, mime_type=mime, system=ZERO_SHOT_SYSTEM,
                                    prompt="Which type is this document?", schema=ZeroShotType).doc_type
        except GeminiError:
            remote = None
        t2 = time.perf_counter()
        rows.append({"file": path.name, "truth": truth, "classifier": local, "gemini": remote,
                     "classifier_ms": round((t1 - t0) * 1000), "gemini_ms": round((t2 - t1) * 1000)})
        print(f"zero-shot {i}/{len(sample)} {truth}: classifier={local} gemini={remote}", flush=True)
        time.sleep(pause)
    return {
        "n": len(rows),
        "classifier_correct": sum(r["classifier"] == r["truth"] for r in rows),
        "gemini_correct": sum(r["gemini"] == r["truth"] for r in rows),
        "gemini_errors": sum(r["gemini"] is None for r in rows),
        "classifier_median_ms": statistics.median(r["classifier_ms"] for r in rows),
        "gemini_median_ms": statistics.median(r["gemini_ms"] for r in rows),
        "per_class": {c: {"classifier": sum(r["classifier"] == c for r in rows if r["truth"] == c),
                          "gemini": sum(r["gemini"] == c for r in rows if r["truth"] == c),
                          "n": sum(r["truth"] == c for r in rows)} for c in CLASSES},
        "rows": rows,
    }


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"


def markdown(rec: dict, zs: dict, model: str) -> str:
    lines = [f"# Gemini evaluation\n", f"{date.today().isoformat()} · `{model}` (free tier)\n",
             "## Receipt extraction — CORD v2 test set\n",
             f"Real receipt photos with hand-labelled fields (Indonesian rupiah), never used for training.\n",
             "| Check | Result |", "|---|---|",
             f"| Local classifier says \"receipt\" | {pct(rec['classified_as_receipt'], rec['n'])} |",
             f"| Total amount exactly right | {pct(rec['total_exact'], rec['with_true_total'])} |",
             f"| Number of line items exactly right | {pct(rec['items_exact'], rec['n'])} |",
             f"| Number of line items within ±1 | {pct(rec['items_within_1'], rec['n'])} |",
             f"| API errors | {rec['errors']} |",
             f"\nMedian time per receipt (classify + extract): {rec['median_latency_ms'] / 1000:.1f} s\n",
             "## Trained classifier vs. asking Gemini\n",
             f"The same {zs['n']} test pages ({zs['n'] // len(CLASSES)} per type), classified by the local model "
             "and by Gemini with no examples (zero-shot).\n",
             "| | Trained classifier | Gemini zero-shot |", "|---|---|---|",
             f"| Accuracy | {pct(zs['classifier_correct'], zs['n'])} | {pct(zs['gemini_correct'], zs['n'])} |",
             f"| Median time per page | {zs['classifier_median_ms']:.0f} ms (CPU) | {zs['gemini_median_ms']:.0f} ms (API) |",
             "| Cost | free, runs offline | API call per page |", "",
             "| Type | Classifier | Gemini |", "|---|---|---|"]
    for c, v in zs["per_class"].items():
        lines.append(f"| {LABELS[c]} | {v['classifier']}/{v['n']} | {v['gemini']}/{v['n']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", type=int, default=100)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between API calls (free-tier limits)")
    ap.add_argument("--rescore", action="store_true", help="re-score the saved answers in gemini.json (no API calls)")
    args = ap.parse_args()
    s = get_settings()
    if args.rescore:
        saved = json.loads((HERE / "gemini.json").read_text(encoding="utf-8"))
        rec, zs = rescore(saved["receipts"]), saved["zero_shot"]
        (HERE / "gemini.json").write_text(json.dumps({"receipts": rec, "zero_shot": zs}, indent=1), encoding="utf-8")
        (HERE / "gemini.md").write_text(markdown(rec, zs, s.vision_model), encoding="utf-8")
        print(f"rescored: total exact {pct(rec['total_exact'], rec['with_true_total'])}")
        return
    sorter = DocSorter(s.model_path, build_gemini(s), s.min_confidence)
    rec = eval_receipts(sorter, args.receipts, args.pause)
    zs = eval_zero_shot(sorter, args.per_class, args.pause)
    (HERE / "gemini.json").write_text(json.dumps({"receipts": rec, "zero_shot": zs}, indent=1), encoding="utf-8")
    (HERE / "gemini.md").write_text(markdown(rec, zs, s.vision_model), encoding="utf-8")
    print(f"\nreceipts: total exact {pct(rec['total_exact'], rec['with_true_total'])}; "
          f"zero-shot: classifier {pct(zs['classifier_correct'], zs['n'])} vs gemini {pct(zs['gemini_correct'], zs['n'])}")


if __name__ == "__main__":
    main()
