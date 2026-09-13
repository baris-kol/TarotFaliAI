"""Günlük burç yorumunu üretip `public/` altına yazar.

    python -m horoscope                       # varsayılan gün (aşağıya bak)
    python -m horoscope --date 2026-09-14     # belirli bir gün
    python -m horoscope --force               # o gün zaten varsa yeniden üret
    python -m horoscope --dry-run             # Gemini'ye gitmeden istemi yazdır

Varsayılan gün: Türkiye saatiyle 18:00'den sonra yarın, önce bugün. Gece
23:17'deki zamanlanmış koşu yarını, 02:47'deki yedek koşu bugünü hedefliyor
(bugün zaten varsa hiçbir şey yapmıyor).

Akış: 12 burç tek istekte üretiliyor; yapısı bozuksa (eksik burç, yarım
metin) baştan, en fazla `GENERATION_ATTEMPTS` kez. Üslup kurallarını
çiğneyen burçlar ise yalnızca kendileri, sorunları söylenerek en fazla
`REPAIR_ROUNDS` tur yeniden yazdırılıyor.

Ortam değişkenleri:
    GEMINI_API_KEY   zorunlu (--dry-run hariç)
    GEMINI_MODEL     isteğe bağlı, virgülle ayrılmış model listesi
    HOROSCOPE_DATE   --date ile aynı (Actions'taki elle çalıştırma kutusu)
    HOROSCOPE_FORCE  "true" ise --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from . import feed, gemini
from .log import info, log, set_output
from .prompt import SYSTEM, build_prompt, build_repair_prompt, response_schema
from .signs import SIGNS
from .sky import TURKEY, Sky, compute_sky

PUBLIC = Path(__file__).resolve().parent.parent / "public"
GENERATION_ATTEMPTS = 3
REPAIR_ROUNDS = 2
EVENING_HOUR = 18


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    day: date = args.date or _default_day()
    root: Path = args.out
    set_output("date", day.isoformat())

    if not args.force and not args.dry_run and feed.has_day(root, day):
        info(f"{day} yorumu zaten var; yeniden üretilmiyor (zorlamak için --force).")
        feed.write_feed(root)
        return 0

    sky = compute_sky(day)
    prompt = build_prompt(sky)
    if args.dry_run:
        info("=== SİSTEM TALİMATI ===\n" + SYSTEM + "\n=== İSTEM ===\n" + prompt)
        return 0

    # Gemini'ye gitmeden önce: yazamayacağımız bir klasör için çağrı harcanmasın.
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        log(
            "error",
            f"Çıktı klasörü oluşturulamadı: {root} ({error}). Ortam değişkeni "
            "cmd'de %TEMP%, PowerShell'de $env:TEMP diye yazılır.",
        )
        return 1

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log(
            "error",
            "GEMINI_API_KEY tanımlı değil. Repo ayarlarında "
            "Secrets and variables → Actions altına ekle.",
        )
        return 1

    models = gemini.models_from_env()
    info(f"{day} için yorum üretiliyor. Model sırası: {', '.join(models)}")
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            reply = gemini.generate(
                api_key=api_key,
                models=models,
                system=SYSTEM,
                prompt=prompt,
                schema=response_schema(),
            )
            readings = feed.parse_readings(reply.text)
            break
        except (gemini.IncompleteReply, feed.InvalidReadings) as error:
            log("warning", f"Deneme {attempt}/{GENERATION_ATTEMPTS} kullanılamadı: {error}")
        except gemini.GeminiError as error:
            log("error", str(error))
            return 1
    else:
        log(
            "error",
            f"{GENERATION_ATTEMPTS} denemede de geçerli yorum alınamadı. "
            "Önceki günler yayında kalıyor.",
        )
        return 1

    readings = _repair(api_key, models, sky, readings)
    path = feed.write_day(root, sky, readings, reply.model)
    days = feed.write_feed(root)
    log("notice", f"{path.name} yazıldı ({reply.model}). gunluk.json: {', '.join(days)}")
    return 0


def _repair(
    api_key: str,
    models: list[str],
    sky: Sky,
    readings: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Üslup kurallarını çiğneyen burçları yeniden yazdırır.

    Turların sonunda hâlâ sorun varsa metin uyarıyla yayınlanıyor: kusurlu
    bir yorum, hiç yorum olmamasından iyi. Uyarı Actions özetinde görünüyor.
    """
    for round_ in range(1, REPAIR_ROUNDS + 1):
        issues = feed.content_issues(readings)
        if not issues:
            return readings
        log("warning", f"Düzeltme turu {round_}/{REPAIR_ROUNDS} — {_summary(readings, issues)}")
        signs = [s for s in SIGNS if s.slug in issues]
        try:
            reply = gemini.generate(
                api_key=api_key,
                models=models,
                system=SYSTEM,
                prompt=build_repair_prompt(sky, readings, issues),
                schema=response_schema(signs),
            )
            fixed = feed.parse_readings(reply.text, expected=signs)
        except (gemini.IncompleteReply, feed.InvalidReadings) as error:
            log("warning", f"Düzeltme turu {round_} kullanılamadı: {error}")
            continue
        except gemini.GeminiError as error:
            log("warning", f"Düzeltme yapılamadı: {error}")
            break
        readings = {slug: fixed.get(slug, entry) for slug, entry in readings.items()}

    issues = feed.content_issues(readings)
    if issues:
        log("warning", f"Kurallara tam uymayan metinlerle yayınlanıyor — {_summary(readings, issues)}")
    return readings


def _summary(readings: dict[str, dict[str, str]], issues: dict[str, list[str]]) -> str:
    return " | ".join(
        f"{readings[slug]['ad']}: {', '.join(problems)}" for slug, problems in issues.items()
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m horoscope")
    parser.add_argument("--date", type=_parse_date, default=_env_date())
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("HOROSCOPE_FORCE", "").strip().lower() == "true",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=PUBLIC)
    return parser.parse_args(argv)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"tarih YYYY-MM-DD olmalı: {value!r}") from None


def _env_date() -> date | None:
    value = os.environ.get("HOROSCOPE_DATE", "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        log("error", f"HOROSCOPE_DATE YYYY-MM-DD olmalı: {value!r}")
        sys.exit(2)


def _default_day() -> date:
    now = datetime.now(TURKEY)
    return now.date() + timedelta(days=1) if now.hour >= EVENING_HOUR else now.date()


if __name__ == "__main__":
    sys.exit(main())
