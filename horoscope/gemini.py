from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .log import info, log

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
)

REQUEST_TIMEOUT = 420
MIN_CALL_SECONDS = 90
OVERLOAD_WAITS = (30, 90, 180, 300)
MAX_RATE_WAIT = 120


class GeminiError(Exception):
    pass


class IncompleteReply(Exception):
    pass


class Postponed(Exception):
    def __init__(self, reason: str, *, quota: bool) -> None:
        super().__init__(reason)
        self.quota = quota


class _Overloaded(Exception):
    pass


class _RateLimited(Exception):
    def __init__(self, message: str, delay: float) -> None:
        super().__init__(message)
        self.delay = delay


class _DailyQuota(Exception):
    pass


class _Missing(Exception):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    requested: str


_spent: dict[str, str] = {}


def models_from_env() -> list[str]:
    raw = os.environ.get("GEMINI_MODEL", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(DEFAULT_MODELS)


def generate(
    *,
    api_key: str,
    models: list[str],
    system: str,
    prompt: str,
    schema: dict,
    deadline: float,
    patient: bool = True,
) -> Reply:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    waits = iter(OVERLOAD_WAITS)
    last = "yanıt yok"
    while True:
        rate_wait = 0.0
        for model in models:
            if model in _spent:
                continue
            remaining = deadline - time.monotonic()
            if remaining < MIN_CALL_SECONDS:
                raise Postponed(f"koşunun süresi doldu ({last})", quota=False)
            try:
                return _call(model, api_key, payload, min(REQUEST_TIMEOUT, remaining))
            except _Overloaded as error:
                last = f"{model}: {error}"
                log("warning", f"{model} şu an yanıt veremiyor ({error}).")
            except _RateLimited as error:
                last = f"{model}: dakikalık sınır"
                rate_wait = max(rate_wait, error.delay)
                log("warning", f"{model} dakikalık sınıra takıldı; {error.delay:.0f} sn beklenecek.")
            except _DailyQuota as error:
                _spent[model] = "kota"
                last = f"{model}: günlük kota doldu"
                log("warning", f"{model} günlük kotası doldu; bu koşuda bir daha denenmeyecek. ({error})")
            except _Missing as error:
                _spent[model] = "yok"
                last = f"{model}: {error}"
                log("warning", f"{model} kullanılamıyor ({error}); bu koşuda bir daha denenmeyecek.")

        if all(model in _spent for model in models):
            quota = any(_spent.get(model) == "kota" for model in models)
            raise Postponed(
                "modellerin günlük kotası doldu" if quota else f"model yok ({last})",
                quota=quota,
            )
        if not patient:
            raise Postponed(f"modeller yoğun ({last})", quota=False)
        wait = max(rate_wait, next(waits, OVERLOAD_WAITS[-1]))
        if time.monotonic() + wait + MIN_CALL_SECONDS > deadline:
            raise Postponed(f"modeller yoğun ve koşunun süresi yetmiyor ({last})", quota=False)
        info(f"Modeller yoğun; {wait:.0f} sn sonra yeniden denenecek.")
        time.sleep(wait)


def _call(model: str, api_key: str, payload: bytes, timeout: float) -> Reply:
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        },
    )
    info(f"{model} çağrılıyor…")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        body = _error_body(error)
        message = _message(body) or str(error.reason)
        if status in (401, 403) or "API key" in message or "API_KEY" in message:
            raise GeminiError(f"Anahtar reddedildi (HTTP {status}): {message}") from None
        if status == 429:
            if _is_daily(body, message):
                raise _DailyQuota(message) from None
            raise _RateLimited(message, _retry_delay(body)) from None
        if status in (400, 404):
            raise _Missing(f"HTTP {status}: {message}") from None
        if status >= 500:
            raise _Overloaded(f"HTTP {status}") from None
        raise GeminiError(f"HTTP {status}: {message}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
        raise _Overloaded(f"bağlantı sorunu: {error}") from None
    return _reply(model, raw)


def _reply(model: str, raw: bytes) -> Reply:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise IncompleteReply("API yanıtı JSON değil") from None

    feedback = data.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise IncompleteReply(f"istem engellendi: {feedback['blockReason']}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise IncompleteReply("yanıtta aday yok")
    candidate = candidates[0]
    reason = candidate.get("finishReason", "STOP")
    if reason != "STOP":
        raise IncompleteReply(f"yanıt tamamlanmadı ({reason})")

    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise IncompleteReply("yanıt boş")

    version = data.get("modelVersion") or model
    usage = data.get("usageMetadata") or {}
    log(
        "notice",
        f"{version}: {usage.get('promptTokenCount', '?')} girdi, "
        f"{usage.get('candidatesTokenCount', '?')} çıktı, "
        f"{usage.get('thoughtsTokenCount', 0)} düşünme token'ı.",
    )
    return Reply(text=text, model=version, requested=model)


def _error_body(error: urllib.error.HTTPError) -> dict:
    try:
        body = json.loads(error.read())
    except Exception:
        return {}
    return body.get("error", {}) if isinstance(body, dict) else {}


def _message(body: dict) -> str:
    return str(body.get("message", "")).strip()


def _details(body: dict, kind: str) -> list[dict]:
    return [
        d for d in body.get("details") or []
        if isinstance(d, dict) and str(d.get("@type", "")).endswith(kind)
    ]


def _is_daily(body: dict, message: str) -> bool:
    for detail in _details(body, "QuotaFailure"):
        for violation in detail.get("violations") or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return "per day" in message.lower()


def _retry_delay(body: dict) -> float:
    for detail in _details(body, "RetryInfo"):
        try:
            return min(float(str(detail.get("retryDelay", "")).rstrip("s")), MAX_RATE_WAIT)
        except ValueError:
            pass
    return 60.0
