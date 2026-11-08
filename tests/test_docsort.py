"""Offline tests: an untrained model and a fake Gemini, so no download, key or network is needed."""
import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from PIL import Image

from docsort.api import create_app
from docsort.classes import CLASSES
from docsort.config import Settings
from docsort.extract import DocumentSummary, ReceiptFields
from docsort.gemini import GeminiError
from docsort.model import build_model
from docsort.report import build_report, safe_cell
from docsort.service import DocSorter, InvalidDocument, load_page


class FakeGemini:
    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, []

    def read_image(self, *, image, mime_type, system, prompt, schema):
        self.calls.append(schema.__name__)
        if self.fail:
            raise GeminiError("quota", 429)
        if schema is ReceiptFields:
            return ReceiptFields(merchant="=HYPERLINK(\"http://evil\")", total=60000,
                                 items=[{"name": "Coffee", "quantity": 2, "price": 40000}])
        return schema.model_validate({"summary": "A test document."}) if schema is DocumentSummary else schema()


@pytest.fixture(scope="session")
def model_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model") / "tiny.pt"
    model = build_model(len(CLASSES), "mobilenet_v3_large", pretrained=False)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "arch": "mobilenet_v3_large", "image_size": 96}, path)
    return path


def png(color=(255, 255, 255), size=(120, 160), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


# ---------------------------------------------------------------- input handling
def test_images_and_pdfs_are_accepted():
    page, data, mime = load_page(png())
    assert page.size == (120, 160) and mime == "image/jpeg" and data[:2] == b"\xff\xd8"
    page, _, _ = load_page(png(fmt="PDF"))  # PIL writes a one-page PDF
    assert page.width > 0


@pytest.mark.parametrize("bad", [b"hello world", b"%PDF-broken", b"<svg onload=alert(1)>", b"\x00" * 100])
def test_junk_is_rejected(bad):
    with pytest.raises(InvalidDocument):
        load_page(bad)


def test_gif_is_rejected():
    with pytest.raises(InvalidDocument):
        load_page(png(fmt="GIF"))


# ---------------------------------------------------------------- classifier + extraction
def test_classification_shape(model_path):
    c = DocSorter(model_path).classify(Image.new("RGB", (100, 140), "white"))
    assert c.doc_type in CLASSES and 0 <= c.confidence <= 1 and len(c.top3) == 3
    probs = [p for _, p in c.top3]
    assert probs == sorted(probs, reverse=True) and probs[0] == c.confidence and sum(probs) <= 1.0001


def test_extraction_failure_is_reported_not_raised(model_path):
    r = DocSorter(model_path, FakeGemini(fail=True)).process(png(), "a.png")
    assert r.fields is None and r.extraction_error


def test_schema_follows_document_type(model_path):
    sorter, gem = DocSorter(model_path), FakeGemini()
    sorter.gemini = gem
    sorter.extract(b"x", "image/jpeg", "receipt")
    sorter.extract(b"x", "image/jpeg", "letter")
    assert gem.calls == ["ReceiptFields", "DocumentSummary"]


# ---------------------------------------------------------------- Excel report
def test_formula_injection_is_neutralised():
    assert safe_cell("=1+1") == "'=1+1" and safe_cell("@cmd") == "'@cmd" and safe_cell("Coffee") == "Coffee"
    assert safe_cell(-5) == -5  # numbers stay numbers


def test_report_has_documents_and_items(model_path):
    sorter = DocSorter(model_path, FakeGemini())
    sorter.classify = lambda page: sorter.__class__.classify(sorter, page).model_copy(update={"doc_type": "receipt"})
    wb = load_workbook(io.BytesIO(build_report([sorter.process(png(), "r1.png")])))
    assert wb.sheetnames == ["Documents", "Receipt items"]
    docs = list(wb["Documents"].values)
    assert docs[1][0] == "r1.png" and docs[1][7] == 60000
    assert str(docs[1][5]).startswith("'=")  # merchant text from the image cannot run as a formula
    assert list(wb["Receipt items"].values)[1][2] == "Coffee"


# ---------------------------------------------------------------- API
@pytest.fixture
def client(model_path):
    settings = Settings(gemini_api_key=None, model_path=model_path, max_file_mb=1, max_files_per_batch=3, rate_limit_per_minute=6)
    with TestClient(create_app(settings, sorter=DocSorter(model_path, FakeGemini()))) as c:
        yield c


def test_classify_and_process(client):
    r = client.post("/classify", files={"file": ("../../etc/passwd.png", png(), "image/png")})
    assert r.status_code == 200 and r.json()["fields"] is None
    assert r.json()["filename"] == "passwd.png", "path parts are stripped from file names"
    r = client.post("/process", files={"file": ("doc.png", png(), "image/png")})
    assert r.status_code == 200 and r.json()["fields"] is not None


def test_upload_limits(client):
    assert client.post("/classify", files={"file": ("big.png", b"0" * (1024 * 1024 + 1), "image/png")}).status_code == 413
    assert client.post("/classify", files={"file": ("empty.png", b"", "image/png")}).status_code == 422
    assert client.post("/classify", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 422
    many = [("files", (f"{i}.png", png(), "image/png")) for i in range(4)]
    assert client.post("/report", files=many).status_code == 422


def test_report_endpoint_returns_xlsx(client):
    r = client.post("/report", files=[("files", ("a.png", png(), "image/png")), ("files", ("b.png", png(), "image/png"))])
    assert r.status_code == 200 and r.content[:2] == b"PK"  # xlsx is a zip file
    assert "attachment" in r.headers["content-disposition"]


def test_rate_limit(client):
    codes = [client.post("/classify", files={"file": ("a.png", png(), "image/png")}).status_code for _ in range(7)]
    assert codes[-1] == 429 and codes[:6] == [200] * 6
