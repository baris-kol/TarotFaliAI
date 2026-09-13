"""Günün gökyüzü — yorumların dayanacağı gerçek konumlar.

Gökyüzünü Gemini'ye sormuyoruz: kendi hâline bırakılınca olmayan bir Merkür
retrosu ya da yanlış bir Ay burcu uydurabiliyor. Bir astroloğun günlük yorum
yazarken baktığı efemeris burada hesaplanıp istemin içine hazır cümleler
olarak giriyor; talimat modelin yalnızca bunları kullanmasını istiyor:

* Güneş, Ay, Merkür–Satürn ve kuşak gezegenleri (Uranüs, Neptün, Plüton):
  burç, gün içi burç geçişi, retro.
* Ay'ın gün içinde tam olan büyük açıları, saatiyle — günlük yorumların asıl
  malzemesi. Ay her gezegenden hızlı olduğu için günde birkaç tane oluyor.
* Gezegenler arası etkin açılar: öğlen `PLANET_ORB` dereceden yakın olanlar,
  yaklaşıyor mu ayrılıyor mu, bugün tam oluyorsa saati.
* Retro dönüşleri: `STATION_WINDOW` içinde başlayan ya da biten retrolar.
* Tutulmalar: `ECLIPSE_WINDOW` içindeki Yeni Ay ve Dolunaylarda Güneş, Ay ve
  Dünya gölgesinin geometrisinden; türüyle (tam / halkalı / parçalı / yarı
  gölge).

Uygulamadaki "Gökyüzü şu an" kartı aynı Güneş/Ay burcunu kendi hesabıyla
gösteriyor — metinle çelişmiyorlar.

Gün, Türkiye saatiyle 00:00–24:00. Türkiye 2016'dan beri yaz saati
uygulamıyor, sabit UTC+3 — saat dilimi veritabanına gerek yok.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import combinations

import ephem

from .signs import Sign, sign_of

TURKEY = timezone(timedelta(hours=3), "TRT")

MONTHS = (
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
)

_SUN = ("Güneş", ephem.Sun)
_MOON = ("Ay", ephem.Moon)
_PLANETS = (
    ("Merkür", ephem.Mercury),
    ("Venüs", ephem.Venus),
    ("Mars", ephem.Mars),
    ("Jüpiter", ephem.Jupiter),
    ("Satürn", ephem.Saturn),
)
_OUTER = (
    ("Uranüs", ephem.Uranus),
    ("Neptün", ephem.Neptune),
    ("Plüton", ephem.Pluto),
)
_OUTER_NAMES = frozenset(name for name, _ in _OUTER)

ASPECTS = (
    ("kavuşum", 0),
    ("altmışlık", 60),
    ("kare", 90),
    ("üçgen", 120),
    ("karşıt", 180),
)

# Gezegenler arası açının "etkin" sayıldığı sapma (derece). Hızlı gezegenler
# için yaklaşık bir iki gün, yavaşlar için birkaç hafta.
PLANET_ORB = 1.5

# (kaç gün öncesinden, kaç gün sonrasına)
STATION_WINDOW = (3, 7)
ECLIPSE_WINDOW = (3, 14)

_PHASE_EVENTS = (
    ("Yeni Ay", ephem.next_new_moon),
    ("İlk Dördün", ephem.next_first_quarter_moon),
    ("Dolunay", ephem.next_full_moon),
    ("Son Dördün", ephem.next_last_quarter_moon),
)

_EARTH_RADIUS_AU = 6378.137 / 149_597_870.7
_PRECISION = timedelta(seconds=30)


@dataclass(frozen=True)
class Ingress:
    """Gün içinde başka bir burca geçiş."""

    sign: Sign
    at: datetime  # Türkiye saati


@dataclass(frozen=True)
class Body:
    name: str
    sign: Sign  # günün başında (00:00)
    ingress: Ingress | None
    retrograde: bool  # öğlen

    def to_json(self) -> dict:
        data: dict = {"burc": self.sign.name, "gecis": None}
        if self.ingress:
            data["gecis"] = {"burc": self.ingress.sign.name, "saat": _hhmm(self.ingress.at)}
        return data


@dataclass(frozen=True)
class Aspect:
    first: str
    second: str
    kind: str
    exact: datetime | None  # bugün tam olduğu an (Türkiye saati)
    orb: float | None = None  # öğlendeki sapma — yalnızca gezegen açıları
    applying: bool | None = None

    @property
    def label(self) -> str:
        return f"{self.first}–{self.second} {self.kind}"

    def involves(self, name: str) -> bool:
        return name in (self.first, self.second)


@dataclass(frozen=True)
class Station:
    """Retro dönüşü: gezegenin görünür hareket yönünü değiştirdiği gün."""

    planet: str
    retrograde: bool  # True: geri hareket başlıyor, False: retro bitiyor
    day: date
    sign: Sign

    @property
    def label(self) -> str:
        return (
            "geri harekete başlıyor (retro başlıyor)"
            if self.retrograde
            else "düz harekete dönüyor (retro bitiyor)"
        )


@dataclass(frozen=True)
class Eclipse:
    solar: bool
    kind: str  # tam / halkalı / parçalı / yarı gölge
    at: datetime  # tam Yeni Ay ya da Dolunay anı (Türkiye saati)
    sign: Sign
    degree: int

    @property
    def name(self) -> str:
        return "Güneş tutulması" if self.solar else "Ay tutulması"


@dataclass(frozen=True)
class PhaseEvent:
    name: str
    at: datetime  # Türkiye saati


@dataclass(frozen=True)
class Sky:
    day: date
    sun: Body
    moon: Body
    planets: tuple[Body, ...]  # Merkür–Satürn
    outer: tuple[Body, ...]  # Uranüs, Neptün, Plüton
    phase: str  # öğlen, uygulamadaki MoonPhase adlarıyla
    illumination: int  # öğlen, yüzde
    event: PhaseEvent | None  # gün içinde tam Yeni Ay / Dördün / Dolunay
    moon_aspects: tuple[Aspect, ...]
    planet_aspects: tuple[Aspect, ...]
    stations: tuple[Station, ...]
    eclipses: tuple[Eclipse, ...]

    @property
    def retrograde(self) -> list[str]:
        return [p.name for p in (*self.planets, *self.outer) if p.retrograde]

    def planet(self, name: str) -> Body | None:
        return next((p for p in (*self.planets, *self.outer) if p.name == name), None)

    def aspects_of(self, name: str) -> list[Aspect]:
        return [a for a in (*self.moon_aspects, *self.planet_aspects) if a.involves(name)]

    def describe(self) -> str:
        """İstemdeki gökyüzü bölümleri."""
        lines = ["GÜNÜN GÖKYÜZÜ (hesaplanmış gerçek konumlar — yalnızca bunları kullan):"]
        lines += [f"- {_body_line(self.sun)}", f"- {_body_line(self.moon)}"]
        lines.append(f"- Ay evresi (öğlen): {self.phase}, aydınlık oranı %{self.illumination}.")
        if self.event:
            lines.append(f"- Bugün tam {self.event.name}: saat {_hhmm(self.event.at)}.")
        lines += [f"- {_body_line(p)}" for p in self.planets]
        lines.append("- Kuşak gezegenleri (yavaş hareket eder, arka plan etkisi):")
        lines += [f"  - {_body_line(p)}" for p in self.outer]
        retro = self.retrograde
        lines.append(
            "- Geri harekette (retro) olan gezegenler: " + ", ".join(retro) + "."
            if retro
            else "- Bugün geri harekette (retro) olan gezegen yok."
        )

        lines += ["", "AY'IN GÜN İÇİNDEKİ AÇILARI (tam oldukları saat, Türkiye saati):"]
        lines += [f"- {_hhmm(a.exact)} {a.label}" for a in self.moon_aspects if a.exact] or [
            "- Bugün Ay'ın tam olan büyük açısı yok."
        ]

        lines += ["", f"GEZEGEN AÇILARI (öğlen itibarıyla {_decimal(PLANET_ORB)}°'lik etki aralığında):"]
        lines += [_planet_aspect_line(a) for a in self.planet_aspects] or [
            "- Etki aralığında gezegen açısı yok."
        ]

        past, ahead = STATION_WINDOW
        lines += ["", f"RETRO DÖNÜŞLERİ ({past} gün öncesinden {ahead} gün sonrasına):"]
        lines += [
            f"- {s.planet}: {day_label(s.day)} tarihinde {s.label}, "
            f"{s.sign.name} burcunda ({_relative(self.day, s.day)})."
            for s in self.stations
        ] or ["- Bu aralıkta başlayan ya da biten retro yok."]

        past, ahead = ECLIPSE_WINDOW
        lines += ["", f"TUTULMALAR ({past} gün öncesinden {ahead} gün sonrasına):"]
        lines += [
            f"- {e.name} ({e.kind}): {day_label(e.at.date())} saat {_hhmm(e.at)}, "
            f"{e.sign.name} {e.degree}° ({_relative(self.day, e.at.date())})."
            for e in self.eclipses
        ] or ["- Bu aralıkta tutulma yok."]
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Yayınlanan dosyadaki `gokyuzu` — yorum hangi gökyüzüne göre yazıldı."""
        return {
            "gunes": self.sun.to_json(),
            "ay": {
                **self.moon.to_json(),
                "evre": self.phase,
                "aydinlik": self.illumination,
            },
            "olay": (
                {"ad": self.event.name, "saat": _hhmm(self.event.at)}
                if self.event
                else None
            ),
            "gezegenler": {
                p.name: {**p.to_json(), "retro": p.retrograde}
                for p in (*self.planets, *self.outer)
            },
            "ayAcilari": [
                {"gezegen": a.second, "aci": a.kind, "saat": _hhmm(a.exact)}
                for a in self.moon_aspects
                if a.exact
            ],
            "gezegenAcilari": [
                {
                    "gezegenler": [a.first, a.second],
                    "aci": a.kind,
                    "fark": round(a.orb or 0, 1),
                    "yaklasiyor": a.applying,
                    "saat": _hhmm(a.exact) if a.exact else None,
                }
                for a in self.planet_aspects
            ],
            "retroDonusleri": [
                {
                    "gezegen": s.planet,
                    "retro": s.retrograde,
                    "tarih": s.day.isoformat(),
                    "burc": s.sign.name,
                }
                for s in self.stations
            ],
            "tutulmalar": [
                {
                    "tur": e.name,
                    "cins": e.kind,
                    "tarih": e.at.date().isoformat(),
                    "saat": _hhmm(e.at),
                    "burc": e.sign.name,
                    "derece": e.degree,
                }
                for e in self.eclipses
            ],
        }


def compute_sky(day: date) -> Sky:
    start = datetime.combine(day, time(0), TURKEY)
    end = start + timedelta(days=1)
    noon = start + timedelta(hours=12)

    elongation = (longitude(ephem.Moon, noon) - longitude(ephem.Sun, noon)) % 360
    return Sky(
        day=day,
        sun=_body(*_SUN, start, end),
        moon=_body(*_MOON, start, end),
        planets=tuple(_body(name, cls, start, end) for name, cls in _PLANETS),
        outer=tuple(_body(name, cls, start, end) for name, cls in _OUTER),
        phase=_phase_name(elongation),
        illumination=round((1 - math.cos(math.radians(elongation))) / 2 * 100),
        event=_phase_event(start, end),
        moon_aspects=_moon_aspects(start),
        planet_aspects=_planet_aspects(start),
        stations=_stations(day),
        eclipses=_eclipses(start),
    )


def longitude(body_class: type, when: datetime) -> float:
    """Görünür, tropikal, jeosantrik ekliptik boylam (derece) — astrolojideki.

    Nutasyon ve sapınç (aberration) dahil: ephem'in ekinoks ve dolunay
    anlarıyla birebir tutuyor. Astrometrik konum (`ephem.Ecliptic(body)`)
    ~0,003° sapıyordu — Güneş'in burç geçişinde 4 dakika.
    """
    moment = _ephem_date(when)
    body = body_class()
    body.compute(moment, epoch=moment)
    apparent = ephem.Equatorial(body.g_ra, body.g_dec, epoch=moment)
    return math.degrees(ephem.Ecliptic(apparent, epoch=moment).lon) % 360


def day_label(day: date) -> str:
    """"16 Eylül"."""
    return f"{day.day} {MONTHS[day.month - 1]}"


def _body(name: str, body_class: type, start: datetime, end: datetime) -> Body:
    sign = sign_of(longitude(body_class, start))
    last = sign_of(longitude(body_class, end))
    ingress = None
    if last != sign:
        ingress = Ingress(last, _ingress_time(body_class, sign, start, end))
    retrograde = _speed(body_class, start + timedelta(hours=12)) < 0
    return Body(name=name, sign=sign, ingress=ingress, retrograde=retrograde)


def _moon_aspects(start: datetime) -> tuple[Aspect, ...]:
    hours = [start + timedelta(hours=h) for h in range(25)]
    moon = [longitude(ephem.Moon, t) for t in hours]
    found: list[Aspect] = []
    for name, cls in (_SUN, *_PLANETS, *_OUTER):
        other = [longitude(cls, t) for t in hours]
        for kind, angle in ASPECTS:
            for target in {angle, -angle % 360}:
                for i in range(24):
                    # Ay her gezegenden hızlı: fark hep artıyor, tam an
                    # eksiden artıya geçiş (±180 sarması artıdan eksiye).
                    if _offset(moon[i], other[i], target) < 0 <= _offset(
                        moon[i + 1], other[i + 1], target
                    ):
                        exact = _root(
                            lambda t, c=cls, g=target: _offset(
                                longitude(ephem.Moon, t), longitude(c, t), g
                            ),
                            hours[i],
                            hours[i + 1],
                        )
                        found.append(Aspect("Ay", name, kind, exact))
    return tuple(sorted(found, key=lambda a: a.exact))


def _planet_aspects(start: datetime) -> tuple[Aspect, ...]:
    end = start + timedelta(days=1)
    noon = start + timedelta(hours=12)
    later = noon + timedelta(hours=6)
    found: list[Aspect] = []
    for (a, cls_a), (b, cls_b) in combinations((_SUN, *_PLANETS, *_OUTER), 2):
        # Kuşak gezegenlerinin kendi aralarındaki açılar yıllarca sürüyor.
        if a in _OUTER_NAMES and b in _OUTER_NAMES:
            continue
        separation = abs(_offset(longitude(cls_a, noon), longitude(cls_b, noon), 0))
        separation_later = abs(_offset(longitude(cls_a, later), longitude(cls_b, later), 0))
        for kind, angle in ASPECTS:
            orb = abs(separation - angle)
            if orb > PLANET_ORB:
                continue
            found.append(
                Aspect(
                    a,
                    b,
                    kind,
                    exact=_exact_between(cls_a, cls_b, angle, start, end),
                    orb=orb,
                    applying=abs(separation_later - angle) < orb,
                )
            )
    return tuple(sorted(found, key=lambda x: x.orb or 0))


def _exact_between(cls_a: type, cls_b: type, angle: int, start: datetime, end: datetime) -> datetime | None:
    for target in {angle, -angle % 360}:
        def gap(t: datetime, g: int = target) -> float:
            return _offset(longitude(cls_a, t), longitude(cls_b, t), g)

        first, last = gap(start), gap(end)
        if (first > 0) != (last > 0) and abs(first) < 10 and abs(last) < 10:
            return _root(gap, start, end)
    return None


def _stations(day: date) -> tuple[Station, ...]:
    past, ahead = STATION_WINDOW
    first = datetime.combine(day - timedelta(days=past), time(0), TURKEY)
    marks = [first + timedelta(days=k) for k in range(past + ahead + 2)]
    found: list[Station] = []
    for name, cls in (*_PLANETS, *_OUTER):
        speeds = [_speed(cls, t) for t in marks]
        for i in range(len(marks) - 1):
            if (speeds[i] < 0) != (speeds[i + 1] < 0):
                at = _root(lambda t, c=cls: _speed(c, t), marks[i], marks[i + 1])
                found.append(
                    Station(
                        planet=name,
                        retrograde=speeds[i + 1] < 0,
                        day=at.date(),
                        sign=sign_of(longitude(cls, at)),
                    )
                )
    return tuple(sorted(found, key=lambda s: s.day))


def _eclipses(start: datetime) -> tuple[Eclipse, ...]:
    past, ahead = ECLIPSE_WINDOW
    begin = _ephem_date(start - timedelta(days=past))
    finish = _ephem_date(start + timedelta(days=ahead + 1))
    found: list[Eclipse] = []
    for solar, next_lunation in ((True, ephem.next_new_moon), (False, ephem.next_full_moon)):
        moment = next_lunation(begin)
        while moment < finish:
            eclipse = _classify_eclipse(moment, solar)
            if eclipse:
                found.append(eclipse)
            moment = next_lunation(ephem.Date(moment + 1))
    return tuple(sorted(found, key=lambda e: e.at))


def _classify_eclipse(lunation: ephem.Date, solar: bool) -> Eclipse | None:
    """Yeni Ay'da Güneş, Dolunay'da Ay tutulması var mı; varsa türü.

    Jeosantrik geometri: Güneş tutulmasında Ay ile Güneş'in merkezleri
    arasındaki en küçük açı, Dünya'dan görülebilirlik sınırıyla (Ay ve Güneş
    yarıçapları + paralaks farkı); Ay tutulmasında Ay ile gölgenin merkezi
    (Güneş'in tam karşısı) arasındaki açı, tam gölge ve yarı gölge
    yarıçaplarıyla (Chauvenet'nin %2 atmosfer payı dahil) karşılaştırılıyor.
    """
    best = None
    for step in range(-12, 13):  # tam anın ±3 saati, 15 dakikada bir
        moment = ephem.Date(lunation + step * 15 * ephem.minute)
        sun, moon = ephem.Sun(moment), ephem.Moon(moment)
        if solar:
            separation = ephem.separation(moon, sun)
        else:
            separation = ephem.separation(moon, (sun.ra + math.pi, -sun.dec))
        if best is None or separation < best[0]:
            best = (separation, sun, moon)
    separation, sun, moon = best

    sun_radius = math.radians(sun.size / 3600) / 2
    moon_radius = math.radians(moon.size / 3600) / 2
    moon_parallax = math.asin(_EARTH_RADIUS_AU / moon.earth_distance)
    sun_parallax = math.asin(_EARTH_RADIUS_AU / sun.earth_distance)

    if solar:
        if separation >= moon_parallax - sun_parallax + moon_radius + sun_radius:
            return None
        if separation < moon_parallax - sun_parallax:
            kind = "tam" if moon_radius > sun_radius else "halkalı"
        else:
            kind = "parçalı"
    else:
        umbra = 1.02 * (moon_parallax + sun_parallax - sun_radius)
        penumbra = 1.02 * (moon_parallax + sun_parallax + sun_radius)
        if separation + moon_radius < umbra:
            kind = "tam"
        elif separation - moon_radius < umbra:
            kind = "parçalı"
        elif separation - moon_radius < penumbra:
            kind = "yarı gölge"
        else:
            return None

    at = _to_turkey(lunation)
    moon_longitude = longitude(ephem.Moon, at)
    return Eclipse(
        solar=solar,
        kind=kind,
        at=at,
        sign=sign_of(moon_longitude),
        degree=int(moon_longitude % 30),
    )


def _ingress_time(body_class: type, sign: Sign, start: datetime, end: datetime) -> datetime:
    """İkiye bölerek burç sınırının geçildiği an (yarım dakika hassasiyet)."""
    low, high = start, end
    while high - low > _PRECISION:
        middle = low + (high - low) / 2
        if sign_of(longitude(body_class, middle)) == sign:
            low = middle
        else:
            high = middle
    return high.astimezone(TURKEY)


def _root(f: Callable[[datetime], float], low: datetime, high: datetime) -> datetime:
    """`f`'nin işaret değiştirdiği an, ikiye bölerek (yarım dakika hassasiyet)."""
    positive = f(low) > 0
    while high - low > _PRECISION:
        middle = low + (high - low) / 2
        if (f(middle) > 0) == positive:
            low = middle
        else:
            high = middle
    return high.astimezone(TURKEY)


def _offset(first: float, second: float, target: float) -> float:
    """`first − second`'in `target` açısından farkı, −180..180 aralığına katlanmış."""
    return (first - second - target + 180) % 360 - 180


def _speed(body_class: type, when: datetime) -> float:
    """Günlük görünür hareket (derece/gün); eksi = geri hareket."""
    half = timedelta(hours=12)
    return _offset(longitude(body_class, when + half), longitude(body_class, when - half), 0)


def _phase_name(elongation: float) -> str:
    # Uygulamadaki MoonPhase eşikleriyle birebir (sky_today.dart).
    for limit, name in (
        (10, "Yeni Ay"),
        (80, "Büyüyen Hilal"),
        (100, "İlk Dördün"),
        (170, "Büyüyen Ay"),
        (190, "Dolunay"),
        (260, "Küçülen Ay"),
        (280, "Son Dördün"),
        (350, "Küçülen Hilal"),
    ):
        if elongation < limit:
            return name
    return "Yeni Ay"


def _phase_event(start: datetime, end: datetime) -> PhaseEvent | None:
    begin, finish = _ephem_date(start), _ephem_date(end)
    for name, next_event in _PHASE_EVENTS:
        moment = next_event(begin)
        if moment < finish:
            return PhaseEvent(name, _to_turkey(moment))
    return None


def _ephem_date(when: datetime) -> ephem.Date:
    return ephem.Date(when.astimezone(timezone.utc).replace(tzinfo=None))


def _to_turkey(moment: ephem.Date) -> datetime:
    return moment.datetime().replace(tzinfo=timezone.utc).astimezone(TURKEY)


def _planet_aspect_line(aspect: Aspect) -> str:
    text = (
        f"- {aspect.label}: fark {_decimal(aspect.orb or 0)}°, "
        f"{'yaklaşıyor' if aspect.applying else 'ayrılıyor'}"
    )
    if aspect.exact:
        text += f", tam olduğu saat {_hhmm(aspect.exact)}"
    return text + "."


def _body_line(body: Body) -> str:
    text = f"{body.name}: {body.sign.name} burcunda"
    if body.retrograde:
        text += ", geri harekette (retro)"
    if body.ingress:
        text += f"; {body.ingress.sign.name} burcuna geçiş saati {_hhmm(body.ingress.at)}"
    return text + "."


def _relative(today: date, other: date) -> str:
    days = (other - today).days
    if days == 0:
        return "bugün"
    if days == 1:
        return "yarın"
    if days == -1:
        return "dün"
    return f"{days} gün sonra" if days > 0 else f"{-days} gün önce"


def _decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _hhmm(moment: datetime) -> str:
    # Saniyeleri yuvarla: 14:31:50 → 14:32.
    return (moment + timedelta(seconds=30)).strftime("%H:%M")
