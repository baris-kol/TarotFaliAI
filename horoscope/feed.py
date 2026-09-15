from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .prompt import FIELDS
from .signs import SIGNS, Sign, find_sign
from .sky import PeriodSky, Sky

FEED_VERSION = 1
FEED_DAYS = 3

LIMITS: dict[str, tuple[int, int]] = {
    "ozet": (15, 180),
    "genel": (250, 1100),
    "askIliskide": (70, 500),
    "askYalniz": (70, 500),
    "kariyer": (90, 600),
    "saglik": (60, 500),
}

PERIOD_LIMITS: dict[str, dict[str, tuple[int, int]]] = {
    "weekly": {
        "ozet": (15, 200),
        "genel": (450, 1500),
        "askIliskide": (120, 650),
        "askYalniz": (120, 650),
        "kariyer": (140, 750),
        "saglik": (100, 600),
    },
    "monthly": {
        "ozet": (15, 210),
        "genel": (600, 1900),
        "askIliskide": (170, 800),
        "askYalniz": (170, 800),
        "kariyer": (190, 900),
        "saglik": (130, 700),
    },
}

PERIOD_FOLDERS = {"weekly": "haftalik", "monthly": "aylik"}
PERIOD_FEED_COUNT = 2

_ARCHIVE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")
_MARKDOWN = re.compile(r"[*#`]+")
_SPACE = re.compile(r"\s+")

_HEALTH_TERMS = (
    r"omu?z", r"sırt", r"omurga", r"boğaz", r"ses tel", r"mide", r"sindirim",
    r"bağırsak", r"karaciğer", r"böbrek", r"akciğer", r"kemik", r"cil[dt]",
    r"kas(?:lar\w*|ın\w*)?(?!\w)", r"sinir sistem", r"bağışıklık", r"hormon",
    r"ağrı", r"sızı", r"baş dön", r"migren", r"tansiyon", r"ödem", r"hastal",
    r"enfeksiyon", r"iltihap", r"alerji", r"şifa",
    r"gıda", r"besin", r"beslen", r"diyet", r"sıvı", r"bol su", r"su iç",
    r"su tüket", r"bitki çay", r"çay(?:ı|lar)?(?!\w)", r"vitamin", r"takviye",
    r"ilaç", r"tedavi", r"terapi", r"detoks", r"kafein",
)
_HEALTH = re.compile(r"(?<!\w)(?:" + "|".join(_HEALTH_TERMS) + r")\w*", re.IGNORECASE)
_MOON_OPENING = re.compile(r"(?:bugün\W+)?(?:gökyüzündeki\s+)?ay(?!\w)", re.IGNORECASE)
_IMPERATIVE = re.compile(r"\w+m[ae]l[ıi]s[ıi]n(?:[ıi]z)?(?!\w)", re.IGNORECASE)

_FOREIGN = re.compile(
    r"(?<!\w)(?:energy|focus|vibe|mood|mindset|timing|healing|journey|feedback|deadline)\w*"
    r"|(?<!\w)(?:the|and|with|your|you|feel|self)(?!\w)"
    r"|\w*[qwx]\w*",
    re.IGNORECASE,
)

_SINGLE_HINT = re.compile(
    r"(?<!\w)(?:yalnızsan|yalnız isen|bekarsan|bekârsan|ilişkin yoksa|kalbin boşsa)",
    re.IGNORECASE,
)
_PARTNER_HINT = re.compile(
    r"(?<!\w)(?:partnerinle|sevgilinle|eşinle|ilişkin varsa|mevcut ilişkin)",
    re.IGNORECASE,
)

REPEAT_FIELDS = ("ozet", "genel")
REPEAT_WORDS = 3
REPEAT_LIMIT = 2
_FILLER = frozenset(
    "bir ve ile için bu da de daha çok sana seni senin sen olarak gibi kadar "
    "en her o şu ki ya ama ancak hem ne gün bugün günün".split()
)
_WORD = re.compile(r"\w+(?:['’]\w+)?")
_SENTENCE = re.compile(r"[.!?;:]")
_ASTRO = re.compile(
    r"\w+['’]\w*|gezegen\w*|yönetici\w*|açı(?:sı|yla|lar\w*)?|kare|üçgen\w*"
    r"|karşıt\w*|kavuşum\w*|altmışlık\w*|retro\w*"
    r"|güneş|ay|merkür|venüs|mars|jüpiter|satürn|uranüs|neptün|plüton"
    r"|koç|boğa|ikizler|yengeç|aslan|başak|terazi|akrep|yay|oğlak|kova|balık"
)


class InvalidReadings(Exception):
    pass


def parse_readings(
    text: str,
    expected: Sequence[Sign] = SIGNS,
    limits: dict[str, tuple[int, int]] = LIMITS,
) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidReadings(f"JSON okunamadı: {error}") from None
    items = data.get("burclar") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise InvalidReadings("yanıtta 'burclar' listesi yok")

    readings: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            problems.append("listede nesne olmayan öğe")
            continue
        sign = find_sign(item.get("burc"))
        if sign is None or sign not in expected:
            problems.append(f"beklenmeyen burç {item.get('burc')!r}")
            continue
        if sign.slug in readings:
            problems.append(f"{sign.name} iki kez yazılmış")
            continue
        entry = {"ad": sign.name}
        for field in FIELDS:
            value = clean_text(item.get(field))
            low, high = limits[field]
            if not low <= len(value) <= high:
                problems.append(f"{sign.name}.{field} {len(value)} karakter ({low}–{high} bekleniyor)")
            entry[field] = value
        readings[sign.slug] = entry

    missing = [s.name for s in expected if s.slug not in readings]
    if missing:
        problems.append("eksik burç: " + ", ".join(missing))
    if problems:
        raise InvalidReadings("; ".join(problems))
    return {s.slug: readings[s.slug] for s in expected}


_CLICHE = re.compile(
    r"harika bir (?:gün|dönem|zaman|fırsat)\w*|son derece|muazzam\w*"
    r"|elverişli bir (?:zaman|dönem|gün)\w*|parlak fikir\w*|tazeleyici"
    r"|fırsatlarla karşılaş\w*|pozitif enerji\w*|yeni kapılar\w*|ilham kayna\w*",
    re.IGNORECASE,
)

ECHO_LIMIT = 0.2
_ECHO_STOP = frozenset(
    "olabilir edebilir yapabilir sağlayabilir bugün günün gününe sana senin "
    "kendi kendini içinde daha fazla kadar olarak üzerine".split()
)


def _stems(entry: dict[str, str]) -> set[str]:
    text = " ".join(entry.get(f, "") for f in ("ozet", "genel"))
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return {
        word[:6]
        for word in _WORD.findall(text)
        if len(word) >= 5 and word not in _ECHO_STOP and not _ASTRO.fullmatch(word)
    }


def echo(entry: dict[str, str], previous: dict[str, str]) -> float:
    today, yesterday = _stems(entry), _stems(previous)
    if not today or not yesterday:
        return 0.0
    return len(today & yesterday) / len(today | yesterday)


def previous_day(root: Path, day: date) -> dict[str, dict[str, str]] | None:
    try:
        data = json.loads(
            archive_path(root, day - timedelta(days=1)).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    readings = data.get("burclar")
    return readings if isinstance(readings, dict) and readings else None


def content_issues(
    readings: dict[str, dict[str, str]],
    previous: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    repeated = _repeated_phrases(readings)
    for slug, entry in readings.items():
        problems: list[str] = []
        cliches = sorted({m.group(0).lower() for f in FIELDS for m in _CLICHE.finditer(entry[f])})
        if cliches:
            problems.append(
                "klişe ifade: " + ", ".join(cliches) + " — somut bir durum ya da imgeyle anlat"
            )
        before = (previous or {}).get(slug)
        if isinstance(before, dict) and echo(entry, before) > ECHO_LIMIT:
            problems.append(
                "dünkü yorumuna çok benziyor (aynı tema ve kelimeler) — Ay'ın bugünkü "
                "konumundan yola çıkarak başka bir açılış, imge ve öneriyle yaz"
            )
        health = sorted({m.group(0).lower() for m in _HEALTH.finditer(entry["saglik"])})
        if health:
            problems.append(
                "sağlık bölümünde organ, belirti, yiyecek ya da içecek var: " + ", ".join(health)
            )
        if _MOON_OPENING.match(entry["genel"]):
            problems.append('genel yorum "Ay" ile başlıyor')
        if _IMPERATIVE.search(entry["ozet"]):
            problems.append('özette "-malısın / -melisin" kalıbı var')
        count = sum(len(_IMPERATIVE.findall(entry[field])) for field in FIELDS)
        if count > 1:
            problems.append(f'"-malısın / -melisin" kalıbı {count} kez geçiyor (en fazla 1)')
        foreign = sorted({m.group(0) for f in FIELDS for m in _FOREIGN.finditer(entry[f])})
        if foreign:
            problems.append("Türkçe olmayan kelime: " + ", ".join(foreign))
        if _SINGLE_HINT.search(entry["askIliskide"]):
            problems.append("askIliskide partneri olmayan okura sesleniyor")
        if _PARTNER_HINT.search(entry["askYalniz"]):
            problems.append("askYalniz partneri olan okura sesleniyor")
        if slug in repeated:
            phrases = ", ".join(f'"{p}"' for p in repeated[slug][:3])
            problems.append(f"başka burçlarda da geçen kalıp ifade: {phrases} — başka kelimelerle anlat")
        if problems:
            issues[slug] = problems
    return issues


def _repeated_phrases(readings: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for slug, entry in readings.items():
        for phrase in _phrases(entry):
            owners.setdefault(phrase, []).append(slug)
    flagged: dict[str, list[str]] = {}
    for phrase, slugs in sorted(owners.items()):
        for slug in slugs[REPEAT_LIMIT:]:
            flagged.setdefault(slug, []).append(phrase)
    return flagged


def _phrases(entry: dict[str, str]) -> set[str]:
    found: set[str] = set()
    for field in REPEAT_FIELDS:
        text = entry[field].replace("İ", "i").replace("I", "ı").lower()
        for sentence in _SENTENCE.split(text):
            words = _WORD.findall(sentence)
            for i in range(len(words) - REPEAT_WORDS + 1):
                gram = words[i : i + REPEAT_WORDS]
                if sum(word in _FILLER for word in gram) >= 2:
                    continue
                if any(_ASTRO.fullmatch(word) for word in gram):
                    continue
                found.add(" ".join(gram))
    return found


def clean_text(value: object) -> str:
    text = _EMOJI.sub("", str(value or ""))
    text = _MARKDOWN.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    if len(text) >= 2 and text[0] in "\"“«" and text[-1] in "\"”»":
        text = text[1:-1].strip()
    return text


def archive_path(root: Path, day: date) -> Path:
    return root / "v1" / "arsiv" / f"{day.isoformat()}.json"


def feed_path(root: Path) -> Path:
    return root / "v1" / "gunluk.json"


def has_day(root: Path, day: date) -> bool:
    try:
        data = json.loads(archive_path(root, day).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return len(data.get("burclar") or {}) == len(SIGNS)


def write_day(root: Path, sky: Sky, readings: dict[str, dict[str, str]], model: str) -> Path:
    record = {
        "tarih": sky.day.isoformat(),
        "uretildi": _now(),
        "model": model,
        "gokyuzu": sky.to_json(),
        "burclar": readings,
    }
    path = archive_path(root, sky.day)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, record)
    return path


def write_feed(root: Path) -> list[str]:
    folder = root / "v1" / "arsiv"
    files = sorted(
        (p for p in folder.glob("*.json") if _ARCHIVE_NAME.match(p.name)),
        key=lambda p: p.name,
        reverse=True,
    )[:FEED_DAYS]
    days = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    path = feed_path(root)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = None
    if not (current and current.get("surum") == FEED_VERSION and current.get("gunler") == days):
        _write_json(path, {"surum": FEED_VERSION, "guncellendi": _now(), "gunler": days})
    return [d["tarih"] for d in days]


def period_path(root: Path, kind: str, start: date) -> Path:
    return root / "v1" / PERIOD_FOLDERS[kind] / f"{start.isoformat()}.json"


def period_feed_path(root: Path, kind: str) -> Path:
    return root / "v1" / f"{PERIOD_FOLDERS[kind]}.json"


def has_period(root: Path, kind: str, start: date) -> bool:
    try:
        data = json.loads(period_path(root, kind, start).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return len(data.get("burclar") or {}) == len(SIGNS)


def write_period(
    root: Path,
    psky: PeriodSky,
    readings: dict[str, dict[str, str]],
    model: str,
) -> Path:
    record = {
        "baslangic": psky.start.isoformat(),
        "bitis": psky.end.isoformat(),
        "uretildi": _now(),
        "model": model,
        "gokyuzu": psky.to_json(),
        "burclar": readings,
    }
    path = period_path(root, psky.kind, psky.start)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, record)
    return path


def write_period_feed(root: Path, kind: str) -> list[str]:
    folder = root / "v1" / PERIOD_FOLDERS[kind]
    files = sorted(
        (p for p in folder.glob("*.json") if _ARCHIVE_NAME.match(p.name)),
        key=lambda p: p.name,
        reverse=True,
    )[:PERIOD_FEED_COUNT]
    periods = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    path = period_feed_path(root, kind)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = None
    if not (
        current
        and current.get("surum") == FEED_VERSION
        and current.get("donemler") == periods
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"surum": FEED_VERSION, "guncellendi": _now(), "donemler": periods})
    return [p["baslangic"] for p in periods]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
