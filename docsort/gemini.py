"""Small Gemini client: image + instructions in, Pydantic-validated JSON out. Retries and a fallback model."""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
RETRYABLE = {429, 500, 502, 503, 504}


class GeminiError(RuntimeError):
    """The AI service failed (bad key, quota, network). Safe to show a generic message."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class Gemini:
    def __init__(self, api_key: str, model: str, fallback_model: str | None = None, max_retries: int = 2):
        from google import genai
        from google.genai import errors, types

        self.types, self.errors = types, errors
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        self.models = [model] + ([fallback_model] if fallback_model else [])
        self.max_retries = max_retries

    def _once(self, model: str, contents, config):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except self.errors.APIError as e:
                code = getattr(e, "code", None)
                if code in RETRYABLE and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt * 2, 20))
                    continue
                raise GeminiError(f"Gemini API error {code}", code) from e
        raise GeminiError("Gemini API: retries exhausted")

    def read_image(self, *, image: bytes, mime_type: str, system: str, prompt: str, schema: type[T]) -> T:
        t = self.types
        contents = [t.Part.from_bytes(data=image, mime_type=mime_type), prompt]
        config = t.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                         response_mime_type="application/json", response_schema=schema)
        for i, model in enumerate(self.models):
            try:
                response = self._once(model, contents, config)
                break
            except GeminiError as e:
                if e.code in RETRYABLE and i + 1 < len(self.models):
                    log.warning("gemini fallback", extra={"fields": {"from": model, "to": self.models[i + 1]}})
                    continue
                raise
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text or "")
        except Exception as e:
            raise GeminiError("model returned an invalid structured response") from e


def build_gemini(settings) -> Gemini:
    if settings.gemini_api_key is None:
        raise GeminiError("GEMINI_API_KEY is not set (copy .env.example to .env and add your key)")
    return Gemini(settings.gemini_api_key.get_secret_value(), settings.vision_model, settings.fallback_vision_model)
