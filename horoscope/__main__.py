from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from . import feed, gemini
from .log import info, log, set_output
from .prompt import (
    FIELD_SPECS,
    PERIOD_FIELD_SPECS,
    SYSTEM,
    build_period_prompt,
    build_period_repair_prompt,
    build_prompt,
    build_repair_prompt,
    period_system,
    response_schema,
)
from .signs import SIGNS
from .sky import TURKEY, compute_period, compute_sky

PUBLIC = Path(__file__).resolve().parent.parent / "public"
GENERATION_ATTEMPTS = 3
REPAIR_ROUNDS = 2
EVENING_HOUR = 18
PERIODS = ("daily", "weekly", "monthly")

Readings = dict[str, dict[str, str]]


@dataclass(frozen=True)
class Job:

    label: str
    exists: bool
    system: str
    prompt: str
    specs: dict[str, str]
    limits: dict[str, tuple[int, int]]
    repair_prompt: Callable[[Readings, dict[str, list[str]]], str]
    write: Callable[[Readings, str], str]
    refresh_feed: Callable[[], object]
    previous: Readings | None = None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    day: date = args.date or _default_day()
    set_output("date", day.isoformat())

    periods = PERIODS if args.period == "all" else (args.period,)
    failed = [period for period in periods if _run(_job(period, day, args.out), args) != 0]
    if failed:
        log("error", "Üretilemeyen: " + ", ".join(failed) + ". Önceki yorumlar yayında kalıyor.")
        return 1
    return 0


def _job(period: str, day: date, root: Path) -> Job:
    if period == "daily":
        sky = compute_sky(day)
        yesterday = compute_sky(day - timedelta(days=1))
        previous = feed.previous_day(root, day)

        def write_day(readings: Readings, model: str) -> str:
            path = feed.write_day(root, sky, readings, model)
            days = feed.write_feed(root)
            return f"{path.name} yazıldı ({model}). gunluk.json: {', '.join(days)}"

        return Job(
            label=f"{day} günlük",
            exists=feed.has_day(root, day),
            system=SYSTEM,
            prompt=build_prompt(sky, yesterday, previous),
            specs=FIELD_SPECS,
            limits=feed.LIMITS,
            repair_prompt=lambda r, i: build_repair_prompt(sky, r, i, yesterday, previous),
            write=write_day,
            refresh_feed=lambda: feed.write_feed(root),
            previous=previous,
        )

    start, end = _bounds(period, day)
    psky = compute_period(period, start, end)
    name = feed.PERIOD_FOLDERS[period]

    def write_period(readings: Readings, model: str) -> str:
        path = feed.write_period(root, psky, readings, model)
        starts = feed.write_period_feed(root, period)
        return f"{name}/{path.name} yazıldı ({model}). {name}.json: {', '.join(starts)}"

    return Job(
        label=f"{psky.title} ({'haftalık' if period == 'weekly' else 'aylık'})",
        exists=feed.has_period(root, period, start),
        system=period_system(period),
        prompt=build_period_prompt(psky),
        specs=PERIOD_FIELD_SPECS[period],
        limits=feed.PERIOD_LIMITS[period],
        repair_prompt=lambda r, i: build_period_repair_prompt(psky, r, i),
        write=write_period,
        refresh_feed=lambda: feed.write_period_feed(root, period),
    )


def _bounds(period: str, day: date) -> tuple[date, date]:
    if period == "weekly":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6)
    start = day.replace(day=1)
    following = (start + timedelta(days=32)).replace(day=1)
    return start, following - timedelta(days=1)


def _run(job: Job, args: argparse.Namespace) -> int:
    if job.exists and not args.force and not args.dry_run:
        info(f"{job.label} yorumu zaten var; yeniden üretilmiyor (zorlamak için --force).")
        job.refresh_feed()
        return 0

    if args.dry_run:
        info(
            f"=== {job.label.upper()} ===\n=== SİSTEM TALİMATI ===\n{job.system}"
            f"\n=== İSTEM ===\n{job.prompt}\n"
        )
        return 0

    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        log(
            "error",
            f"Çıktı klasörü oluşturulamadı: {args.out} ({error}). Ortam değişkeni "
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
    info(f"{job.label} yorumu üretiliyor. Model sırası: {', '.join(models)}")
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            reply = gemini.generate(
                api_key=api_key,
                models=models,
                system=job.system,
                prompt=job.prompt,
                schema=response_schema(SIGNS, job.specs),
            )
            readings = feed.parse_readings(reply.text, limits=job.limits)
            break
        except (gemini.IncompleteReply, feed.InvalidReadings) as error:
            log("warning", f"{job.label}: deneme {attempt}/{GENERATION_ATTEMPTS} kullanılamadı: {error}")
        except gemini.GeminiError as error:
            log("error", f"{job.label}: {error}")
            return 1
    else:
        log("error", f"{job.label}: {GENERATION_ATTEMPTS} denemede de geçerli yorum alınamadı.")
        return 1

    readings = _repair(api_key, models, job, readings)
    log("notice", job.write(readings, reply.model))
    return 0


def _repair(api_key: str, models: list[str], job: Job, readings: Readings) -> Readings:
    for round_ in range(1, REPAIR_ROUNDS + 1):
        issues = feed.content_issues(readings, job.previous)
        if not issues:
            return readings
        log("warning", f"{job.label}: düzeltme turu {round_}/{REPAIR_ROUNDS} — {_summary(readings, issues)}")
        signs = [s for s in SIGNS if s.slug in issues]
        try:
            reply = gemini.generate(
                api_key=api_key,
                models=models,
                system=job.system,
                prompt=job.repair_prompt(readings, issues),
                schema=response_schema(signs, job.specs),
            )
            fixed = feed.parse_readings(reply.text, expected=signs, limits=job.limits)
        except (gemini.IncompleteReply, feed.InvalidReadings) as error:
            log("warning", f"{job.label}: düzeltme turu {round_} kullanılamadı: {error}")
            continue
        except gemini.GeminiError as error:
            log("warning", f"{job.label}: düzeltme yapılamadı: {error}")
            break
        readings = {slug: fixed.get(slug, entry) for slug, entry in readings.items()}

    issues = feed.content_issues(readings, job.previous)
    if issues:
        log(
            "warning",
            f"{job.label}: kurallara tam uymayan metinlerle yayınlanıyor — "
            f"{_summary(readings, issues)}",
        )
    return readings


def _summary(readings: Readings, issues: dict[str, list[str]]) -> str:
    return " | ".join(
        f"{readings[slug]['ad']}: {', '.join(problems)}" for slug, problems in issues.items()
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m horoscope")
    parser.add_argument("--date", type=_parse_date, default=_env_date())
    parser.add_argument(
        "--period",
        choices=("all", *PERIODS),
        default=os.environ.get("HOROSCOPE_PERIOD", "").strip() or "all",
    )
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
