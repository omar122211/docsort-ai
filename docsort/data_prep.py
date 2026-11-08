"""Download a slice of two public datasets and save them as an image folder:

    data/images/{train,val,test}/{class}/*.jpg

- RVL-CDIP (scanned office documents, 16 types) — we keep 6 types.
- CORD v2 (receipt photos, CC BY 4.0) — the 7th type, "receipt".

Receipts are photos while RVL-CDIP pages are grayscale scans, so receipts are converted to grayscale and
resized the same way. Otherwise the model could learn "photo vs scan" instead of "what kind of document".

    python -m docsort.data_prep            # ~2.5 GB download, a few minutes
"""
from __future__ import annotations

import io
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from docsort.classes import RVL_IDS
from docsort.config import get_settings

RVL_REPO = "chainyo/rvl-cdip"
CORD_REPO = "naver-clova-ix/cord-v2"
# The RVL-CDIP parquet files are sorted by label, so each type sits in its own shards.
# rows per split, number of shards, rows per type (from the dataset card)
RVL_LAYOUT = {"train": (320_000, 119, 20_000), "val": (40_000, 15, 2_500), "test": (40_000, 15, 2_500)}


def rvl_shard(split: str, label: int) -> str:
    """The shard holding the middle of this label's block of rows."""
    rows, shards, per_label = RVL_LAYOUT[split]
    index = int((per_label * label + per_label / 2) // (rows / shards))
    return f"data/{split}-{index:05d}-of-{shards:05d}.parquet"


RVL_SHARDS = {split: sorted({rvl_shard(split, label) for label in RVL_IDS}) for split in RVL_LAYOUT}
CORD_SPLITS = {"train": ["data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
                         "data/train-00001-of-00004-7dbbe248962764c5.parquet",
                         "data/train-00002-of-00004-688fe1305a55e5cc.parquet"],
               "val": ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
               "test": ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"]}
MAX_SIDE = 640  # plenty for a 320-px classifier, keeps the folder small
MAX_PER_CLASS = {"train": 650, "val": 170, "test": 170}


def to_page(raw: bytes) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("L")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def save(img: Image.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=88)


def image_bytes(cell) -> bytes:
    return cell["bytes"] if isinstance(cell, dict) else cell


def main() -> None:
    s = get_settings()
    root = s.data_dir / "images"
    random.seed(7)
    for split in ("train", "val", "test"):
        counts: Counter = Counter()
        for shard in RVL_SHARDS[split]:
            path = hf_hub_download(RVL_REPO, shard, repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
            table = pq.read_table(path)
            for i, (img, label) in enumerate(zip(table.column("image").to_pylist(), table.column("label").to_pylist())):
                name = RVL_IDS.get(label)
                if name is None or counts[name] >= MAX_PER_CLASS[split]:
                    continue
                try:
                    save(to_page(image_bytes(img)), root / split / name / f"rvl-{Path(shard).stem}-{i}.jpg")
                    counts[name] += 1
                except OSError:
                    pass  # a few scans are corrupt — skip them
        for shard in CORD_SPLITS[split]:
            path = hf_hub_download(CORD_REPO, shard, repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
            for i, img in enumerate(pq.read_table(path, columns=["image"]).column("image").to_pylist()):
                if counts["receipt"] >= MAX_PER_CLASS[split]:
                    break
                save(to_page(image_bytes(img)), root / split / "receipt" / f"cord-{Path(shard).stem[:20]}-{i}.jpg")
                counts["receipt"] += 1
        print(split, dict(sorted(counts.items())))
    make_samples()


def make_samples(skip: int = 40) -> None:
    """One full-resolution test page per type for the demo page (samples/)."""
    s = get_settings()
    out = s.data_dir.parent / "samples"
    out.mkdir(exist_ok=True)
    for label, name in RVL_IDS.items():
        path = hf_hub_download(RVL_REPO, rvl_shard("test", label), repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
        table = pq.read_table(path)
        found = [i for i, lab in enumerate(table.column("label").to_pylist()) if lab == label]
        raw = image_bytes(table.column("image")[found[skip]].as_py())
        Image.open(io.BytesIO(raw)).convert("L").save(out / f"{name}.jpg", "JPEG", quality=85)
    path = hf_hub_download(CORD_REPO, CORD_SPLITS["test"][0], repo_type="dataset", cache_dir=s.data_dir / "hf-cache")
    raw = image_bytes(pq.read_table(path, columns=["image"]).column("image")[3].as_py())
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((1400, 1400))
    img.save(out / "receipt.jpg", "JPEG", quality=85)
    print("samples:", sorted(p.name for p in out.glob("*.jpg")))


if __name__ == "__main__":
    main()
