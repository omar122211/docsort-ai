"""FastAPI service.  Run:  uvicorn docsort.api:app"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from docsort.config import Settings, get_settings
from docsort.gemini import Gemini, GeminiError, build_gemini
from docsort.logging_setup import setup_logging
from docsort.report import build_report
from docsort.service import DocSorter, InvalidDocument, Result

log = logging.getLogger("docsort.api")
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class RateLimiter:
    """In-memory sliding window per client (protects the free API quota)."""

    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str, cost: int = 1) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) + cost > self.per_minute:
            return False
        q.extend([now] * cost)
        return True


def create_app(settings: Settings | None = None, gemini: Gemini | None = None, sorter: DocSorter | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    state: dict = {"sorter": sorter, "extraction": gemini is not None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["sorter"] is None:
            g = gemini
            if g is None:
                try:
                    g = build_gemini(settings)
                except GeminiError as e:
                    log.warning("field extraction disabled", extra={"fields": {"reason": str(e)}})
            state["sorter"] = DocSorter(settings.model_path, g, settings.min_confidence)
            state["extraction"] = g is not None
        yield

    app = FastAPI(title="DocSort AI", version="0.1.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    max_bytes = settings.max_file_mb * 1024 * 1024

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, started = uuid.uuid4().hex[:12], time.perf_counter()
        response = await call_next(request)
        log.info("request", extra={"fields": {"id": rid, "method": request.method, "path": request.url.path,
                                               "status": response.status_code, "ms": int((time.perf_counter() - started) * 1000)}})
        response.headers["X-Request-ID"] = rid
        return response

    async def read_upload(file: UploadFile) -> bytes:
        data = await file.read(max_bytes + 1)  # never read more than the limit into memory
        if len(data) > max_bytes:
            raise HTTPException(413, f"{_name(file)} is larger than {settings.max_file_mb} MB")
        if not data:
            raise HTTPException(422, f"{_name(file)} is empty")
        return data

    def check_rate(request: Request, cost: int = 1) -> None:
        if not limiter.allow(request.client.host if request.client else "unknown", cost):
            raise HTTPException(429, "Too many documents — please wait a minute.")

    def run(data: bytes, file: UploadFile, extract: bool) -> Result:
        try:
            return state["sorter"].process(data, _name(file), extract=extract)
        except InvalidDocument as e:
            raise HTTPException(422, f"{_name(file)}: {e}")

    @app.get("/health")
    def health():
        s: DocSorter | None = state["sorter"]
        return {"status": "ok" if s else "starting", "classes": s.classes if s else [],
                "classifier": s.metrics if s else {}, "extraction": state["extraction"]}

    @app.post("/classify", response_model=Result)
    async def classify(request: Request, file: UploadFile = File(...)):
        """Document type only — runs locally, no AI API call."""
        check_rate(request)
        return run(await read_upload(file), file, extract=False)

    @app.post("/process", response_model=Result)
    async def process(request: Request, file: UploadFile = File(...)):
        """Document type + fields (receipt / invoice fields, or a short summary for other types)."""
        check_rate(request)
        return run(await read_upload(file), file, extract=True)

    @app.post("/report")
    async def report(request: Request, files: list[UploadFile] = File(...)):
        """Several documents in, one Excel file out."""
        if len(files) > settings.max_files_per_batch:
            raise HTTPException(422, f"at most {settings.max_files_per_batch} files per report")
        check_rate(request, cost=len(files))
        results = [run(await read_upload(f), f, extract=True) for f in files]
        return Response(build_report(results), media_type=XLSX,
                        headers={"Content-Disposition": 'attachment; filename="docsort-report.xlsx"'})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong."})

    return app


def _name(file: UploadFile) -> str:
    # keep only a safe, short display name — it ends up in JSON and Excel, never on disk
    name = (file.filename or "document").replace("\\", "/").split("/")[-1]
    return "".join(ch for ch in name if ch.isprintable())[:120] or "document"


app = create_app()
