"""Gemini'ye giden talimat, istem ve yanıt şeması.

Yorumların kalitesi büyük ölçüde bu dosyada. Değiştirirken:

* Gökyüzü modelden istenmiyor, `sky.py` hesaplıyor; talimat modelin yalnızca
  verilen konumları kullanmasını istiyor. Bu kuralı gevşetme.
* Alan uzunlukları uygulamadaki mektup düzenine göre. `feed.LIMITS` (günlük)
  ve `feed.PERIOD_LIMITS` (haftalık/aylık) aynı aralıkları (karakter olarak,
  daha geniş) denetliyor — birini değiştirirsen ötekine de bak.
* Üslup kurallarının bir kısmı (sağlıkta organ/yiyecek, "Ay" ile başlayan
  giriş, "-malısın", burçlar arası kalıp tekrarı, İngilizce kelime, aşk
  bölümlerinin doğru okura seslenmesi) `feed.content_issues` ile de
  denetleniyor; çiğneyen
  burçlar `build_repair_prompt` ile yeniden yazdırılıyor. Kuralı burada
  değiştirirsen orada da değiştir.
* Talimat parçalardan kuruluyor: üslup, sınırlar ve çeşitlilik günlük,
  haftalık ve aylık yorumda ortak; yalnızca giriş ve gökyüzü bölümü dönemine
  göre değişiyor.
* Değişikliği Gemini'ye gitmeden görmek için: `python -m horoscope --dry-run`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta

from .signs import HOUSE_THEMES, SIGNS, Sign, house_of
from .sky import PeriodSky, Sky, day_label

FIELD_SPECS: dict[str, str] = {
    "ozet": "Günün tek cümlelik özü; en fazla 110 karakter. Akılda kalan, "
    "merak uyandıran bir cümle; emir kipi yok.",
    "genel": "Günün genel havası; 3–4 cümle, 55–85 kelime, tek paragraf.",
    "askIliskide": "Partneri ya da sevgilisi olan okur için aşk ve ilişki; "
    "2 cümle, 20–40 kelime.",
    "askYalniz": "Partneri olmayan okur için aşk, flört ve yeni tanışmalar; "
    "2 cümle, 20–40 kelime.",
    "kariyer": "İş, okul ve para; 2 cümle, 25–45 kelime.",
    "saglik": "Enerji, dinlenme ve zihin dengesi; 1–2 cümle, 20–35 kelime. "
    "Organ, belirti, yiyecek ya da içecek yok.",
}

FIELDS: tuple[str, ...] = tuple(FIELD_SPECS)

# Haftalık ve aylık yorumda alanlar aynı (uygulama aynı mektup düzeniyle
# gösteriyor), yalnızca uzunluk ve bakış farklı.
PERIOD_FIELD_SPECS: dict[str, dict[str, str]] = {
    "weekly": {
        "ozet": "Haftanın tek cümlelik özü; en fazla 120 karakter. Akılda "
        "kalan bir cümle; emir kipi yok.",
        "genel": "Haftanın genel havası ve akışı; 5–6 cümle, 90–130 kelime, "
        "tek paragraf. Haftanın hangi bölümünde neyin öne çıktığını söyle.",
        "askIliskide": "Partneri ya da sevgilisi olan okur için haftanın aşk "
        "ve ilişki havası; 2–3 cümle, 30–55 kelime.",
        "askYalniz": "Partneri olmayan okur için haftanın aşk, flört ve yeni "
        "tanışmaları; 2–3 cümle, 30–55 kelime.",
        "kariyer": "Haftanın iş, okul ve para gündemi; 2–3 cümle, 35–60 kelime.",
        "saglik": "Haftanın enerji, dinlenme ve zihin dengesi; 2 cümle, 25–45 "
        "kelime. Organ, belirti, yiyecek ya da içecek yok.",
    },
    "monthly": {
        "ozet": "Ayın tek cümlelik özü; en fazla 130 karakter. Akılda kalan "
        "bir cümle; emir kipi yok.",
        "genel": "Ayın genel havası ve akışı; 6–8 cümle, 120–170 kelime, tek "
        "paragraf. Ayın ilk yarısıyla ikinci yarısının farkını anlat.",
        "askIliskide": "Partneri ya da sevgilisi olan okur için ayın aşk ve "
        "ilişki havası; 3 cümle, 40–65 kelime.",
        "askYalniz": "Partneri olmayan okur için ayın aşk, flört ve yeni "
        "tanışmaları; 3 cümle, 40–65 kelime.",
        "kariyer": "Ayın iş, okul ve para gündemi; 3 cümle, 45–75 kelime.",
        "saglik": "Ayın enerji, dinlenme ve zihin dengesi; 2–3 cümle, 30–55 "
        "kelime. Organ, belirti, yiyecek ya da içecek yok.",
    },
}

_ROLE = (
    "Sen, Arkanay adlı Türkçe tarot ve burç uygulaması için {kind} burç "
    "yorumları yazan deneyimli bir astrologsun. Okurların, {rhythm} uygulamayı "
    "açıp kendi burcunun yorumunu okuyan, astrolojiye meraklı ama uzman "
    "olmayan insanlar.\n"
)

_STYLE = """
ÜSLUP
- Modern, doğal ve akıcı bir Türkçe kullan. Çeviri kokan kalıplardan, \
klişelerden ve laf kalabalığından kaçın. Araya İngilizce kelime karıştırma.
- Okura "sen" diye hitap et. Tonun sıcak, sakin ve cesaret verici olsun ama \
gerçekçiliği bırakma.
- Kehanette bulunma. "Kesinlikle", "mutlaka", "kaçınılmaz", "kaderin" gibi \
ifadeler yerine eğilim ve öneri dili kullan: "… için uygun bir gün", \
"… fark edebilirsin", "… iyi gelebilir". Kesin gelecek ("-acaktır") yerine \
olasılık dilini ("-abilir") seç.
- Emir kipiyle konuşma. "-malısın / -melisin" kalıbını özette hiç kullanma, \
bir burcun bütün metninde en fazla bir kez kullan.
- Genel yorumu "Ay", "Bugün Ay" ya da "Bugün gökyüzündeki Ay" diye \
başlatma; her burçta başka bir girişle başla (bir imge, günün duygusu, bir \
soru…).
- Korkutucu, suçlayıcı ya da kaderci cümle kurma. Zorlayıcı bir etkiyi \
anlatırken onu nasıl iyi kullanabileceğini de söyle.
"""

_SKY_DAILY = """
GÖKYÜZÜ
- Yalnızca istemde verilen gökyüzü bilgilerini kullan. Orada yazmayan bir \
gezegen konumu, retro, açı, tutulma ya da gök olayı uydurma. Retro olarak \
listelenmeyen hiçbir gezegenin retro olduğunu söyleme.
- Açıların dili: kavuşum iki gezegenin konularını birleştirip yoğunlaştırır; \
üçgen ve altmışlık akış, kolaylık ve destek getirir; kare gerilim ve harekete \
geçme baskısı yaratır; karşıt denge arayışı ve başkalarıyla yüzleşme \
demektir. Zorlayıcı açıları da nasıl iyi kullanılacağıyla birlikte anlat.
- Her burç için listeden o burca en çok dokunan bir iki etkiyi seç: açıyı \
yapan gezegenlerin o burcun hangi evlerinde olduğuna, yöneticisinin açılarına, \
yaklaşan dönüşlere ve tutulmalara bak. Hepsini sayma; aynı açıyı her burçta \
tekrarlama.
- Günün bir bölümünü ("sabah", "öğleden sonra", "akşama doğru") yalnızca \
listede saati verilen olaylar için (Ay'ın açıları, burç geçişleri, Ay evresi) \
kullan; saati rakamla yazma. Gezegen açıları gün boyu sürer; onlara saat ya \
da günün bir bölümünü yakıştırma.
- Kuşak gezegenlerini (Uranüs, Neptün, Plüton) yalnızca bir açıya ya da \
dönüşe karıştıklarında an.
- Başlayan ya da biten bir retroyu o gezegenin konularıyla ilişkilendir \
(Merkür: iletişim, yolculuk, belgeler; Venüs: ilişkiler, değerler, para; \
Mars: enerji, girişim; Jüpiter: büyüme, inançlar; Satürn: sorumluluk, yapı).
- Yaklaşan ya da yeni geçmiş bir tutulma listelendiyse, tutulmanın o burcun \
hangi evine düştüğünü kullan; tutulmayı korkutucu anlatma, bir kapanış ya da \
yeni başlangıç eşiği olarak anlat.
- Göndermeleri ölçülü tut: bir burcun yorumunda bir iki somut gökyüzü \
göndermesi yeter. "Ev" terimini yığmak yerine çoğu zaman o evin konusunu \
söyle ("Ay ilişkiler alanında dolaşırken" gibi).
- Burç ve gezegen adlarına gelen ekleri ses uyumuna göre doğru yaz \
(Terazi'de, Akrep'te, Yay'da, Oğlak'ta, Merkür'ün).
"""

_SKY_PERIOD = """
GÖKYÜZÜ
- Yalnızca istemde verilen gökyüzü bilgilerini kullan. Orada yazmayan bir \
gezegen konumu, retro, açı, tutulma ya da gök olayı uydurma. Retro olarak \
listelenmeyen hiçbir gezegenin retro olduğunu söyleme.
- Açıların dili: kavuşum iki gezegenin konularını birleştirip yoğunlaştırır; \
üçgen ve altmışlık akış, kolaylık ve destek getirir; kare gerilim ve harekete \
geçme baskısı yaratır; karşıt denge arayışı ve başkalarıyla yüzleşme \
demektir. Zorlayıcı açıları da nasıl iyi kullanılacağıyla birlikte anlat.
- Bu bir {period} yorum: dönemin akışını anlat. Olayları tarih sırasıyla \
sayma; her burç için o burca en çok dokunan iki üç olayı seç ve dönemin \
hangi bölümünde öne çıktığını söyle ("hafta ortasında", "ayın ikinci \
yarısında", "Perşembe'den itibaren").
- Belirli bir güne yalnızca listede tarihi verilen olaylar için gönderme \
yap; günü adıyla ya da tarihiyle yaz ("Salı", "22 Eylül'de"); saati rakamla \
yazma.
- Ay her iki üç günde bir burç değiştirir; Ay'ın geçişlerini ancak bir \
günün havasını anlatırken, ölçülü kullan. Dönemin asıl malzemesi Yeni Ay, \
Dolunay, gezegenlerin burç geçişleri, retro dönüşleri ve gezegen açıları.
- Kuşak gezegenlerini (Uranüs, Neptün, Plüton) yalnızca bir açıya ya da \
dönüşe karıştıklarında an.
- Başlayan ya da biten bir retroyu o gezegenin konularıyla ilişkilendir \
(Merkür: iletişim, yolculuk, belgeler; Venüs: ilişkiler, değerler, para; \
Mars: enerji, girişim; Jüpiter: büyüme, inançlar; Satürn: sorumluluk, yapı).
- Dönemde bir tutulma varsa, tutulmanın o burcun hangi evine düştüğünü \
kullan; tutulmayı korkutucu anlatma, bir kapanış ya da yeni başlangıç eşiği \
olarak anlat.
- Göndermeleri ölçülü tut. "Ev" terimini yığmak yerine çoğu zaman o evin \
konusunu söyle ("Dolunay ilişkiler alanını aydınlatırken" gibi).
- Burç ve gezegen adlarına gelen ekleri ses uyumuna göre doğru yaz \
(Terazi'de, Akrep'te, Yay'da, Oğlak'ta, Merkür'ün).
"""

_LIMITS = """
SINIRLAR
- Sağlık bölümü yalnızca enerji, dinlenme, uyku, hareket, stres ve günün \
ritmi üzerine olsun. Organ ya da beden bölgesi (omuz, sırt, boğaz, mide, \
cilt, kaslar…), belirti ya da hastalık (ağrı, baş dönmesi, ödem…), yiyecek, \
içecek, su, bitki çayı, diyet, ilaç, takviye ya da terapi anma. \
Astrolojideki "her burç bir organı yönetir" geleneğini kullanma. Tıbbi \
tavsiye, teşhis ya da tedavi önerisi verme.
- Kariyer bölümünde yatırım, borsa, kripto, al-sat ya da kesin kazanç vaadi \
olmasın.
- Aşk iki ayrı bölüm: askIliskide yalnızca partneri olan okura, askYalniz \
yalnızca partneri olmayan okura seslensin. Birinde ötekine seslenme \
("yalnızsan…", "partnerinle…" yanlış bölümde olmasın); iki bölüm birbirini \
tekrar etmesin. İkisinde de okurun cinsiyetini ya da ilişki biçimini varsayma.
- Şanslı sayı, şanslı renk, emoji, madde işareti, başlık ya da markdown \
kullanma. Burcun adını metnin içinde tekrar tekrar anma.
"""

_VARIETY = """
ÇEŞİTLİLİK
- On iki yorum birbirinin kopyası gibi okunmasın. Her burca farklı bir \
açılışla, farklı bir imgeyle ve farklı bir öneriyle yaz; aynı cümle kalıbını \
iki burçta kullanma.
- Herkesi etkileyen bir olayı (Ay'ın burç değiştirmesi, aynı açı) anlatırken \
burçtan burca aynı ifadeyi ("sabah saatlerinden itibaren", "güne başlarken" \
gibi) tekrarlama; her burçta başka kelimelerle ve başka bir yerden gir.
- Her burcu kendi elementinin, niteliğinin ve yönetici gezegeninin \
karakteriyle ele al.
"""

SYSTEM = _ROLE.format(kind="günlük", rhythm="her gün") + _STYLE + _SKY_DAILY + _LIMITS + _VARIETY

_PERIOD_WORDS = {
    "weekly": ("haftalık", "her hafta başında"),
    "monthly": ("aylık", "her ayın başında"),
}


def period_system(kind: str) -> str:
    """Haftalık ya da aylık yorumun talimatı."""
    label, rhythm = _PERIOD_WORDS[kind]
    return (
        _ROLE.format(kind=label, rhythm=rhythm)
        + _STYLE
        + _SKY_PERIOD.replace("{period}", label)
        + _LIMITS
        + _VARIETY
    )


_WEEKDAYS = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")
_CLOSING = (
    "\n\nOn iki burcun hepsini Koç'tan Balık'a sırayla yaz. \"burc\" "
    "alanına burcun adını yukarıdaki yazımla koy."
)


def build_prompt(sky: Sky) -> str:
    return _context(sky) + _CLOSING


def build_period_prompt(psky: PeriodSky) -> str:
    return _period_context(psky) + _CLOSING


def build_repair_prompt(
    sky: Sky,
    readings: dict[str, dict[str, str]],
    issues: dict[str, list[str]],
) -> str:
    """Kuralları çiğneyen burçları, sorunlarıyla birlikte yeniden yazdıran istem."""
    return _repair(_context(sky), readings, issues)


def build_period_repair_prompt(
    psky: PeriodSky,
    readings: dict[str, dict[str, str]],
    issues: dict[str, list[str]],
) -> str:
    return _repair(_period_context(psky), readings, issues)


def _repair(
    context: str,
    readings: dict[str, dict[str, str]],
    issues: dict[str, list[str]],
) -> str:
    lines = [
        context,
        "",
        "DÜZELTME — önceki yanıtında aşağıdaki burçlar kurallara uymadı. Bu kez "
        "on iki burcun hepsini değil, yalnızca bu burçları bütün alanlarıyla "
        "yeniden yaz: belirtilen sorunları gider, geri kalan her kurala da uy, "
        "gökyüzü göndermelerini koru. \"burc\" alanına burcun adını yukarıdaki "
        "yazımla koy.",
    ]
    for slug, problems in issues.items():
        entry = readings[slug]
        previous = json.dumps({field: entry[field] for field in FIELDS}, ensure_ascii=False)
        lines += [f"- {entry['ad']}: {'; '.join(problems)}", f"  Önceki metin: {previous}"]
    return "\n".join(lines)


def response_schema(
    signs: Sequence[Sign] = SIGNS,
    specs: dict[str, str] = FIELD_SPECS,
) -> dict:
    entry = {
        "type": "object",
        "properties": {
            "burc": {"type": "string", "enum": [s.name for s in signs]},
            **{
                field: {"type": "string", "description": spec}
                for field, spec in specs.items()
            },
        },
        "required": ["burc", *specs],
    }
    return {
        "type": "object",
        "properties": {
            "burclar": {
                "type": "array",
                "items": entry,
                "minItems": len(signs),
                "maxItems": len(signs),
            },
        },
        "required": ["burclar"],
    }


def _context(sky: Sky) -> str:
    """Tarih, gökyüzü, burç burç vurgular ve alan tarifleri — son talimat hariç."""
    day = sky.day
    lines = [
        f"Tarih: {day_label(day)} {day.year}, "
        f"{_WEEKDAYS[day.weekday()]} (Türkiye saatiyle gün boyu).",
        "",
        sky.describe(),
        "",
        "BURÇ BURÇ GÜNÜN VURGULARI (her burç kendi burcunu 1. ev sayar; "
        "parantez içinde o evin konuları):",
        *(f"- {_sign_line(sign, sky)}" for sign in SIGNS),
        "",
        "YAZILACAK ALANLAR (her burç için):",
        *(f"- {field}: {spec}" for field, spec in FIELD_SPECS.items()),
    ]
    return "\n".join(lines)


def _period_context(psky: PeriodSky) -> str:
    """Dönem, gökyüzü, burç burç vurgular ve alan tarifleri."""
    span = (
        "Pazartesi–Pazar"
        if psky.kind == "weekly"
        else f"{psky.start.day}–{psky.end.day} {day_label(psky.end).split()[1]}"
    )
    lines = [
        f"Dönem: {psky.title} ({span}, Türkiye saatiyle).",
        "",
        psky.describe(),
        "",
        "BURÇ BURÇ DÖNEMİN VURGULARI (her burç kendi burcunu 1. ev sayar; "
        "parantez içinde o evin konuları):",
        *(f"- {_period_sign_line(sign, psky)}" for sign in SIGNS),
        "",
        "YAZILACAK ALANLAR (her burç için):",
        *(f"- {field}: {spec}" for field, spec in PERIOD_FIELD_SPECS[psky.kind].items()),
    ]
    return "\n".join(lines)


def _sign_line(sign: Sign, sky: Sky) -> str:
    notes = [
        _transit("Güneş", sign, sky.sun.sign, sky.sun.ingress),
        _transit("Ay", sign, sky.moon.sign, sky.moon.ingress),
    ]
    # Ay'ın açıları zaten ayrı listede; Yengeç için tekrar sayılmıyor.
    if sign.ruler != "Ay":
        ruler_aspects = [a.label for a in sky.aspects_of(sign.ruler)]
        if ruler_aspects:
            notes.append(f"yöneticisinin bugünkü açıları: {', '.join(ruler_aspects)}")
    ruler = sky.planet(sign.ruler)
    if ruler and ruler.retrograde:
        notes.append(f"yöneticisi {ruler.name} geri harekette")
    for station in sky.stations:
        if station.planet == sign.ruler:
            notes.append(
                f"yöneticisi {station.planet} {day_label(station.day)} tarihinde {station.label}"
            )
    for eclipse in sky.eclipses:
        house = house_of(sign, eclipse.sign)
        notes.append(
            f"{eclipse.name} ({day_label(eclipse.at.date())}) {house}. evinde "
            f"({HOUSE_THEMES[house - 1]})"
        )
    if sky.sun.sign == sign:
        notes.append("Güneş kendi burcunda: yaş günü dönemi")
    return (
        f"{sign.name} ({sign.element} · {sign.modality} · yöneticisi {sign.ruler}): "
        + "; ".join(notes)
        + "."
    )


def _period_sign_line(sign: Sign, psky: PeriodSky) -> str:
    """Bir burç için dönemin öne çıkanları: Güneş'in evi, Yeni Ay / Dolunay /
    tutulmanın düştüğü evler, yöneticisinin geçişleri, retroları ve açıları."""

    def house(other: Sign) -> str:
        number = house_of(sign, other)
        return f"{number}. evinde ({HOUSE_THEMES[number - 1]})"

    notes: list[str] = []
    sun = psky.body("Güneş")
    if sun:
        notes.append(f"Güneş {house(sun.sign)}")
    birthday = sun is not None and sun.sign == sign

    for event in psky.events:
        when = day_label(event.at.date())
        if event.kind == "gecis" and event.bodies == ("Güneş",) and event.sign:
            notes.append(f"Güneş {when} tarihinde {house_of(sign, event.sign)}. evine geçiyor")
            birthday = birthday or event.sign == sign
        elif event.kind in ("evre", "tutulma") and event.phase in ("Yeni Ay", "Dolunay"):
            label = event.text.split(" — ")[-1] if event.kind == "tutulma" else event.phase
            notes.append(f"{label} ({when}) {house(event.sign)}")

    ruler = sign.ruler
    if ruler == "Ay":
        notes.append("yöneticisi Ay: dönemin Yeni Ay ve Dolunay'ı bu burç için daha belirgin")
    elif ruler != "Güneş":
        body = psky.body(ruler)
        if body:
            text = f"yöneticisi {ruler} dönem başında {house(body.sign)}"
            if body.retrograde:
                text += ", geri harekette"
            notes.append(text)
        for event in psky.events:
            if ruler not in event.bodies:
                continue
            when = day_label(event.at.date())
            if event.kind in ("gecis", "retro"):
                notes.append(f"{when}: {event.text}")
            elif event.kind == "aci":
                notes.append(f"{when}: {event.text} (yöneticisinin açısı)")

    if birthday:
        notes.append("Güneş kendi burcunda: yaş günü dönemi")
    return (
        f"{sign.name} ({sign.element} · {sign.modality} · yöneticisi {sign.ruler}): "
        + "; ".join(notes)
        + "."
    )


def _transit(body: str, sign: Sign, at: Sign, ingress) -> str:
    house = house_of(sign, at)
    text = f"{body} {house}. evinde ({HOUSE_THEMES[house - 1]})"
    if ingress:
        later = house_of(sign, ingress.sign)
        clock = (ingress.at + timedelta(seconds=30)).strftime("%H:%M")
        text += f", saat {clock} itibarıyla {later}. evine geçiyor ({HOUSE_THEMES[later - 1]})"
    return text
