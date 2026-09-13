"""Yanıtın denetlenmesi ve yayınlanan dosyalar.

    public/v1/arsiv/YYYY-MM-DD.json   bir günün yorumu — kalıcı arşiv
    public/v1/gunluk.json             uygulamanın okuduğu dosya: arşivdeki
                                      en yeni 3 gün, yeniden eskiye

Uygulama `gunluk.json`'dan cihazın bugününe denk gelen günü seçiyor. Yarının
yorumu gece 23:17'de yayınlandığında bugünkü hâlâ dosyada; Türkiye'den geri
saat dilimlerindeki kullanıcılar da kendi "bugün"lerini buluyor; bir gece
üretim başarısız olursa önceki gün yerinde duruyor.

İki ayrı denetim var:

* `parse_readings` yapıyı denetliyor (her burç bir kez, alanlar makul
  uzunlukta). Bozuksa yanıtın tamamı yeniden üretiliyor.
* `content_issues` üslup kurallarını denetliyor (sağlıkta organ/yiyecek,
  "Ay" ile başlayan giriş, "-malısın", burçlar arası kalıp tekrarı,
  İngilizce kelime, aşk bölümlerinin doğru okura seslenmesi). Çiğneyen
  burçlar yalnızca kendileri yeniden yazdırılıyor.

Format değişirse `v1` klasörünü bozma, `v2` aç: yayındaki eski uygulama
sürümleri `v1`'i okumaya devam ediyor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from .prompt import FIELDS
from .signs import SIGNS, Sign, find_sign
from .sky import Sky

FEED_VERSION = 1
FEED_DAYS = 3

# Karakter sınırları. Alt sınır yarım kalmış metni yakalıyor; üst sınır
# istemdeki kelime aralığının epey üstünde, yalnızca kontrolden çıkmış
# yanıtı eliyor.
LIMITS: dict[str, tuple[int, int]] = {
    "ozet": (15, 180),
    "genel": (250, 1100),
    "askIliskide": (70, 500),
    "askYalniz": (70, 500),
    "kariyer": (90, 600),
    "saglik": (60, 500),
}

_ARCHIVE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
# Emoji blokları, ☀–➿ semboller, varyasyon seçici ve birleştirici.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")
_MARKDOWN = re.compile(r"[*#`]+")
_SPACE = re.compile(r"\s+")

# Sağlık bölümünde istenmeyen konular: organ ve beden bölgeleri, belirti ve
# hastalık, yiyecek, içecek, tedavi. Türkçe ekler için kelime başından
# eşleşiyor; masum kelimelere takılan kökler bilerek yok ("boyun" →
# "boyunca", "eklem" → "eklemek", "göz" → "göz önünde", "kalp" → "kalbinin
# sesi") ya da sınırlı ("kas" → "kasım" değil, "çay" → "çayır" değil).
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

# Modelin arada bir sızdırdığı İngilizce kelimeler ("energyyle") ve Türkçe
# alfabede olmayan q, w, x harfli kelimeler.
_FOREIGN = re.compile(
    r"(?<!\w)(?:energy|focus|vibe|mood|mindset|timing|healing|journey|feedback|deadline)\w*"
    r"|(?<!\w)(?:the|and|with|your|you|feel|self)(?!\w)"
    r"|\w*[qwx]\w*",
    re.IGNORECASE,
)

# Aşk bölümlerinin yanlış okura seslenmesi: ilişkide olanlara "yalnızsan",
# yalnızlara "partnerinle".
_SINGLE_HINT = re.compile(
    r"(?<!\w)(?:yalnızsan|yalnız isen|bekarsan|bekârsan|ilişkin yoksa|kalbin boşsa)",
    re.IGNORECASE,
)
_PARTNER_HINT = re.compile(
    r"(?<!\w)(?:partnerinle|sevgilinle|eşinle|ilişkin varsa|mevcut ilişkin)",
    re.IGNORECASE,
)

# Burçlar arası kalıp tekrarı: özet ve genel yorumdaki üç kelimelik diziler.
# Aynı diziyi en fazla REPEAT_LIMIT burç kullanabiliyor; fazlası (burç
# sırasıyla sonrakiler) düzeltmeye gidiyor. İki ya da üç kelimesi bağlaç,
# zamir gibi dolgu kelimesi olan diziler ("için harika bir") sayılmıyor.
REPEAT_FIELDS = ("ozet", "genel")
REPEAT_WORDS = 3
REPEAT_LIMIT = 2
_FILLER = frozenset(
    "bir ve ile için bu da de daha çok sana seni senin sen olarak gibi kadar "
    "en her o şu ki ya ama ancak hem ne gün bugün günün".split()
)
_WORD = re.compile(r"\w+(?:['’]\w+)?")
_SENTENCE = re.compile(r"[.!?;:]")
# Gökyüzü göndermeleri ("Venüs'ün Plüton ile", "yönetici gezegenin") aynı
# olayı anlatan burçlarda doğal olarak ortak; kalıp sayılmıyor. Kesme
# işaretli her kelime özel ad (burç, gezegen).
_ASTRO = re.compile(
    r"\w+['’]\w*|gezegen\w*|yönetici\w*|açı(?:sı|yla|lar\w*)?|kare|üçgen\w*"
    r"|karşıt\w*|kavuşum\w*|altmışlık\w*|retro\w*"
    r"|güneş|ay|merkür|venüs|mars|jüpiter|satürn|uranüs|neptün|plüton"
    r"|koç|boğa|ikizler|yengeç|aslan|başak|terazi|akrep|yay|oğlak|kova|balık"
)


class InvalidReadings(Exception):
    """Modelin yanıtı yayınlanabilir değil — yeniden üretilmeli."""


def parse_readings(
    text: str,
    expected: Sequence[Sign] = SIGNS,
) -> dict[str, dict[str, str]]:
    """Model yanıtı → `{slug: {ad, ozet, genel, ask, kariyer, saglik}}`, burç sırasıyla.

    `expected`: yanıtta olması gereken burçlar — düzeltme turunda yalnızca
    yeniden yazdırılanlar.
    """
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
            low, high = LIMITS[field]
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


def content_issues(readings: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Üslup kurallarını çiğneyen burçlar: `{slug: [sorunlar]}`; temizse boş."""
    issues: dict[str, list[str]] = {}
    repeated = _repeated_phrases(readings)
    for slug, entry in readings.items():
        problems: list[str] = []
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
    """`REPEAT_LIMIT`'ten fazla burçta geçen diziler → fazladan kullanan burçlar."""
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
        # Python'un lower()'ı Türkçe bilmiyor: "İ" → "i̇", "I" → "i".
        text = entry[field].replace("İ", "i").replace("I", "ı").lower()
        # Diziler cümle sınırını aşmasın ("…kayabilir. Yönetici …").
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
    # Özet bazen tırnak içinde geliyor.
    if len(text) >= 2 and text[0] in "\"“«" and text[-1] in "\"”»":
        text = text[1:-1].strip()
    return text


def archive_path(root: Path, day: date) -> Path:
    return root / "v1" / "arsiv" / f"{day.isoformat()}.json"


def feed_path(root: Path) -> Path:
    return root / "v1" / "gunluk.json"


def has_day(root: Path, day: date) -> bool:
    """O günün eksiksiz yorumu arşivde var mı?"""
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
    """`gunluk.json`'u arşivin en yeni günlerinden yeniden kurar.

    İçerik değişmediyse dosyaya dokunmuyor — yedek koşu boş yere commit
    üretmesin. Dönen değer dosyadaki tarihler.
    """
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


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
