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
WEEKDAYS = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")

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

PLANET_ORB = 1.5

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

    sign: Sign
    at: datetime


@dataclass(frozen=True)
class Body:
    name: str
    sign: Sign
    ingress: Ingress | None
    retrograde: bool

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
    exact: datetime | None
    orb: float | None = None
    applying: bool | None = None

    @property
    def label(self) -> str:
        return f"{self.first}–{self.second} {self.kind}"

    def involves(self, name: str) -> bool:
        return name in (self.first, self.second)


@dataclass(frozen=True)
class Station:

    planet: str
    retrograde: bool
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
    kind: str
    at: datetime
    sign: Sign
    degree: int

    @property
    def name(self) -> str:
        return "Güneş tutulması" if self.solar else "Ay tutulması"


@dataclass(frozen=True)
class PhaseEvent:
    name: str
    at: datetime


@dataclass(frozen=True)
class Sky:
    day: date
    sun: Body
    moon: Body
    planets: tuple[Body, ...]
    outer: tuple[Body, ...]
    phase: str
    illumination: int
    event: PhaseEvent | None
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


@dataclass(frozen=True)
class PeriodEvent:

    at: datetime
    kind: str
    text: str
    phase: str | None = None
    bodies: tuple[str, ...] = ()
    sign: Sign | None = None

    def to_json(self) -> dict:
        data = {
            "tarih": self.at.date().isoformat(),
            "saat": _hhmm(self.at),
            "tur": self.kind,
            "metin": self.text,
        }
        if self.phase:
            data["evre"] = self.phase
        return data


@dataclass(frozen=True)
class PeriodSky:

    kind: str
    start: date
    end: date
    bodies: tuple[Body, ...]
    events: tuple[PeriodEvent, ...]

    @property
    def title(self) -> str:
        if self.kind == "monthly":
            return f"{MONTHS[self.start.month - 1]} {self.start.year}"
        if self.start.month == self.end.month:
            return (
                f"{self.start.day}–{self.end.day} {MONTHS[self.end.month - 1]} "
                f"{self.end.year} haftası"
            )
        return f"{day_label(self.start)} – {day_label(self.end)} {self.end.year} haftası"

    def body(self, name: str) -> Body | None:
        return next((b for b in self.bodies if b.name == name), None)

    def describe(self) -> str:
        lines = [
            "DÖNEMİN GÖKYÜZÜ (hesaplanmış gerçek konumlar ve olaylar — yalnızca bunları kullan):",
            "Dönemin başında konumlar:",
        ]
        for body in self.bodies:
            text = f"- {body.name}: {body.sign.name} burcunda"
            if body.retrograde:
                text += ", geri harekette (retro)"
            lines.append(text + ".")
        retro = [b.name for b in self.bodies if b.retrograde]
        lines.append(
            "- Dönemin başında geri harekette (retro) olanlar: " + ", ".join(retro) + "."
            if retro
            else "- Dönemin başında geri harekette (retro) olan gezegen yok."
        )
        lines += ["", "DÖNEMİN OLAYLARI (Türkiye saatiyle, zaman sırasıyla):"]
        lines += [
            f"- {day_label(e.at.date())} {weekday_label(e.at.date())} {_hhmm(e.at)} — {e.text}."
            for e in self.events
        ] or ["- Bu dönemde listelenecek olay yok."]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "konumlar": {
                b.name: {"burc": b.sign.name, "retro": b.retrograde} for b in self.bodies
            },
            "olaylar": [e.to_json() for e in self.events],
        }


def compute_period(kind: str, start: date, end: date) -> PeriodSky:
    begin = datetime.combine(start, time(0), TURKEY)
    finish = datetime.combine(end + timedelta(days=1), time(0), TURKEY)
    everyone = (_SUN, *_PLANETS, *_OUTER)

    bodies = tuple(_body(name, cls, begin, begin + timedelta(days=1)) for name, cls in everyone)
    events: list[PeriodEvent] = []
    events += _period_phases(begin, finish)
    for name, cls in everyone:
        events += _period_ingresses(name, cls, begin, finish, timedelta(days=1))
    if kind == "weekly":
        events += _period_ingresses(*_MOON, begin, finish, timedelta(hours=6))
    events += _period_stations(begin, finish)
    events += _period_aspects(begin, finish)
    events.sort(key=lambda e: e.at)
    return PeriodSky(kind=kind, start=start, end=end, bodies=bodies, events=tuple(events))


def _period_phases(begin: datetime, finish: datetime) -> list[PeriodEvent]:
    found: list[PeriodEvent] = []
    stop = _ephem_date(finish)
    for name, next_event in _PHASE_EVENTS:
        moment = next_event(_ephem_date(begin))
        while moment < stop:
            at = _to_turkey(moment)
            lon = longitude(ephem.Moon, at)
            sign = sign_of(lon)
            eclipse = None
            if name in ("Yeni Ay", "Dolunay"):
                eclipse = _classify_eclipse(moment, solar=name == "Yeni Ay")
            text = f"{name}, {sign.name} {int(lon % 30)}°"
            if eclipse:
                text += f" — {eclipse.name.lower()} ({eclipse.kind})"
            found.append(
                PeriodEvent(
                    at=at,
                    kind="tutulma" if eclipse else "evre",
                    text=text,
                    phase=name,
                    sign=sign,
                )
            )
            moment = next_event(ephem.Date(moment + 1))
    return found


def _period_ingresses(
    name: str, cls: type, begin: datetime, finish: datetime, step: timedelta
) -> list[PeriodEvent]:
    found: list[PeriodEvent] = []
    t = begin
    sign = sign_of(longitude(cls, t))
    while t < finish:
        nxt = min(t + step, finish)
        later = sign_of(longitude(cls, nxt))
        if later != sign:
            at = _ingress_time(cls, sign, t, nxt)
            backwards = name != "Ay" and _speed(cls, at) < 0
            found.append(
                PeriodEvent(
                    at=at,
                    kind="gecis",
                    text=f"{name} {later.name} burcuna geçiyor"
                    + (" (geri hareketle)" if backwards else ""),
                    bodies=(name,),
                    sign=later,
                )
            )
        t, sign = nxt, later
    return found


def _period_stations(begin: datetime, finish: datetime) -> list[PeriodEvent]:
    found: list[PeriodEvent] = []
    marks = [begin]
    while marks[-1] < finish:
        marks.append(min(marks[-1] + timedelta(days=1), finish))
    for name, cls in (*_PLANETS, *_OUTER):
        speeds = [_speed(cls, t) for t in marks]
        for i in range(len(marks) - 1):
            if (speeds[i] < 0) != (speeds[i + 1] < 0):
                at = _root(lambda t, c=cls: _speed(c, t), marks[i], marks[i + 1])
                sign = sign_of(longitude(cls, at))
                starting = speeds[i + 1] < 0
                found.append(
                    PeriodEvent(
                        at=at,
                        kind="retro",
                        text=(
                            f"{name} {sign.name} burcunda geri harekete başlıyor (retro başlıyor)"
                            if starting
                            else f"{name} {sign.name} burcunda düz harekete dönüyor (retro bitiyor)"
                        ),
                        bodies=(name,),
                        sign=sign,
                    )
                )
    return found


def _period_aspects(begin: datetime, finish: datetime) -> list[PeriodEvent]:
    marks = [begin]
    while marks[-1] < finish:
        marks.append(min(marks[-1] + timedelta(days=1), finish))
    everyone = (_SUN, *_PLANETS, *_OUTER)
    lons = {name: [longitude(cls, t) for t in marks] for name, cls in everyone}

    found: list[PeriodEvent] = []
    for (a, cls_a), (b, cls_b) in combinations(everyone, 2):
        if a in _OUTER_NAMES and b in _OUTER_NAMES:
            continue
        for kind, angle in ASPECTS:
            for target in {angle, -angle % 360}:
                gaps = [_offset(x, y, target) for x, y in zip(lons[a], lons[b])]
                for i in range(len(marks) - 1):
                    first, last = gaps[i], gaps[i + 1]
                    if (first > 0) != (last > 0) and abs(first) < 10 and abs(last) < 10:
                        at = _root(
                            lambda t, ca=cls_a, cb=cls_b, g=target: _offset(
                                longitude(ca, t), longitude(cb, t), g
                            ),
                            marks[i],
                            marks[i + 1],
                        )
                        found.append(
                            PeriodEvent(at=at, kind="aci", text=f"{a}–{b} {kind}", bodies=(a, b))
                        )
    return found


def weekday_label(day: date) -> str:
    return WEEKDAYS[day.weekday()]


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
    moment = _ephem_date(when)
    body = body_class()
    body.compute(moment, epoch=moment)
    apparent = ephem.Equatorial(body.g_ra, body.g_dec, epoch=moment)
    return math.degrees(ephem.Ecliptic(apparent, epoch=moment).lon) % 360


def day_label(day: date) -> str:
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
    best = None
    for step in range(-12, 13):
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
    low, high = start, end
    while high - low > _PRECISION:
        middle = low + (high - low) / 2
        if sign_of(longitude(body_class, middle)) == sign:
            low = middle
        else:
            high = middle
    return high.astimezone(TURKEY)


def _root(f: Callable[[datetime], float], low: datetime, high: datetime) -> datetime:
    positive = f(low) > 0
    while high - low > _PRECISION:
        middle = low + (high - low) / 2
        if (f(middle) > 0) == positive:
            low = middle
        else:
            high = middle
    return high.astimezone(TURKEY)


def _offset(first: float, second: float, target: float) -> float:
    return (first - second - target + 180) % 360 - 180


def _speed(body_class: type, when: datetime) -> float:
    half = timedelta(hours=12)
    return _offset(longitude(body_class, when + half), longitude(body_class, when - half), 0)


def _phase_name(elongation: float) -> str:
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
    return (moment + timedelta(seconds=30)).strftime("%H:%M")
