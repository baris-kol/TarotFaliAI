"""Gemini REST istemcisi.

Bağımlılıksız (yalnızca urllib): Actions'ta kurulacak paket azalsın, SDK
sürümleri arasındaki değişiklikler bu dosyayı bozmasın. Anahtar URL'de değil
`x-goog-api-key` başlığında gidiyor — hata mesajlarına ve loglara karışmasın.
"""

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

# Sırayla denenir: model yoksa (404), bu anahtara kapalıysa ya da günlük
# kotası dolduysa sıradakine geçilir. GEMINI_MODEL değişkeni (virgülle
# ayrılmış liste) bunu ezer — Google bir modeli kapattığında kod değil,
# yalnızca repodaki değişken güncellenir.
DEFAULT_MODELS = (
    "gemini-3.8-flash", 
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)

# 12 burç tek yanıtta ~8–10 bin çıktı token'ı; düşünme payıyla birlikte
# bir iki dakika sürebiliyor.
REQUEST_TIMEOUT = 300

# Geçici hatalarda (kota anlık dolu, sunucu meşgul) bekleyip aynı modeli
# yeniden dene; hepsi tükenirse sıradaki modele geç.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (15, 45, 90)


class GeminiError(Exception):
    """Devam etmenin anlamı olmayan hata: anahtar geçersiz, hiçbir model yok."""


class IncompleteReply(Exception):
    """Model yanıt verdi ama kullanılamaz: yarım kaldı, engellendi ya da boş."""


class _ModelUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    model: str


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
) -> Reply:
    """Yapılandırılmış (JSON şemalı) tek bir yanıt üretir."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    reasons: list[str] = []
    for model in models:
        try:
            return _call(model, api_key, payload)
        except _ModelUnavailable as error:
            reasons.append(f"{model}: {error}")
            log("warning", f"{model} kullanılamadı ({error}).")
    raise GeminiError("Hiçbir model yanıt vermedi — " + " | ".join(reasons))


def _call(model: str, api_key: str, payload: bytes) -> Reply:
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        },
    )

    problem = ""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        info(f"{model} çağrılıyor (deneme {attempt + 1})…")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read()
            return _reply(model, raw)
        except urllib.error.HTTPError as error:
            status, message = error.code, _error_message(error)
            if status in (401, 403) or "API key" in message or "API_KEY" in message:
                raise GeminiError(
                    f"Anahtar reddedildi (HTTP {status}): {message}"
                ) from None
            if status in (400, 404):
                # Model adı yanlış/kapatılmış ya da bu modelde şema desteklenmiyor.
                raise _ModelUnavailable(f"HTTP {status}: {message}") from None
            if status not in TRANSIENT_STATUS:
                raise GeminiError(f"HTTP {status}: {message}") from None
            problem = f"HTTP {status}: {message}"
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as error:
            problem = f"bağlantı sorunu: {error}"

        if attempt < len(BACKOFF_SECONDS):
            wait = BACKOFF_SECONDS[attempt]
            log("warning", f"{model}: {problem} — {wait} sn sonra yeniden denenecek.")
            time.sleep(wait)
    raise _ModelUnavailable(problem)


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
    return Reply(text=text, model=version)


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read())
        return str(body["error"]["message"])
    except Exception:  # noqa: BLE001 — gövde her zaman JSON değil
        return str(error.reason)
