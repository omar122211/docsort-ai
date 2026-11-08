"""DocSort pipeline: validate the upload → classify the page (local model) → extract fields (Gemini)."""
from __future__ import annotations

import io
import logging
import time

import torch
from PIL import Image, ImageOps
from pydantic import BaseModel

from docsort.classes import LABELS
from docsort.extract import SCHEMAS, SYSTEM, DocumentSummary, extraction_prompt
from docsort.gemini import Gemini, GeminiError
from docsort.model import eval_transform, load_checkpoint

log = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse "decompression bomb" images
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}


class InvalidDocument(ValueError):
    """The upload is not a readable image or PDF."""


class Classification(BaseModel):
    doc_type: str
    label: str
    confidence: float
    unsure: bool
    top3: list[tuple[str, float]]


class Result(BaseModel):
    filename: str
    classification: Classification
    fields: dict | None = None
    extraction_error: str | None = None
    latency_ms: int


def load_page(data: bytes) -> tuple[Image.Image, bytes, str]:
    """Return (page image, bytes to send to Gemini, mime type). PDFs: the first page is rendered."""
    if data[:5] == b"%PDF-":
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(data)
            page = pdf[0].render(scale=150 / 72).to_pil()  # 150 dpi
            pdf.close()
        except Exception as e:
            raise InvalidDocument("could not read the PDF") from e
    else:
        try:
            page = Image.open(io.BytesIO(data))
            if page.format not in ALLOWED_FORMATS:
                raise InvalidDocument(f"unsupported image type {page.format}")
            page.load()
        except (OSError, Image.DecompressionBombError) as e:
            raise InvalidDocument("not a readable image") from e
    page = ImageOps.exif_transpose(page).convert("RGB")
    page.thumbnail((2000, 2000))  # enough detail for reading; re-encoding also strips metadata
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=90)
    return page, buf.getvalue(), "image/jpeg"


class DocSorter:
    def __init__(self, model_path, gemini: Gemini | None = None, min_confidence: float = 0.55):
        self.model, ckpt = load_checkpoint(model_path)
        self.classes: list[str] = ckpt["classes"]
        self.transform = eval_transform(ckpt["image_size"])
        self.metrics = ckpt.get("metrics", {})
        self.gemini, self.min_confidence = gemini, min_confidence

    @torch.inference_mode()
    def classify(self, page: Image.Image) -> Classification:
        probs = self.model(self.transform(page).unsqueeze(0)).softmax(1)[0]
        top = probs.topk(3)
        best = self.classes[int(top.indices[0])]
        conf = float(top.values[0])
        return Classification(doc_type=best, label=LABELS[best], confidence=round(conf, 4), unsure=conf < self.min_confidence,
                              top3=[(self.classes[int(i)], round(float(v), 4)) for v, i in zip(top.values, top.indices)])

    def extract(self, image: bytes, mime: str, doc_type: str) -> dict:
        if self.gemini is None:
            raise GeminiError("field extraction is not configured")
        schema = SCHEMAS.get(doc_type, DocumentSummary)
        out = self.gemini.read_image(image=image, mime_type=mime, system=SYSTEM, prompt=extraction_prompt(doc_type), schema=schema)
        return out.model_dump()

    def process(self, data: bytes, filename: str, extract: bool = True) -> Result:
        started = time.perf_counter()
        page, image, mime = load_page(data)
        cls = self.classify(page)
        fields, error = None, None
        if extract:
            try:
                fields = self.extract(image, mime, cls.doc_type)
            except GeminiError as e:
                log.warning("extraction failed", extra={"fields": {"reason": str(e)}})
                error = "Field extraction is unavailable right now."
        latency = int((time.perf_counter() - started) * 1000)
        log.info("processed", extra={"fields": {"type": cls.doc_type, "confidence": cls.confidence,
                                                "extracted": fields is not None, "latency_ms": latency}})
        return Result(filename=filename, classification=cls, fields=fields, extraction_error=error, latency_ms=latency)
