"""Gemini'ye giden talimat, istem ve yanıt şeması.

Yorumların kalitesi büyük ölçüde bu dosyada. Değiştirirken:

* Gökyüzü modelden istenmiyor, `sky.py` hesaplıyor; talimat modelin yalnızca
  verilen konumları kullanmasını istiyor. Bu kuralı gevşetme.
* Alan uzunlukları uygulamadaki mektup düzenine göre. `feed.LIMITS` aynı
  aralıkları (karakter olarak, daha geniş) denetliyor — birini değiştirirsen
  ötekine de bak.
* Üslup kurallarının bir kısmı (sağlıkta organ/yiyecek, "Ay" ile başlayan
  giriş, "-malısın", burçlar arası kalıp tekrarı, İngilizce kelime, aşk
  bölümlerinin doğru okura seslenmesi) `feed.content_issues` ile de
  denetleniyor; çiğneyen
  burçlar `build_repair_prompt` ile yeniden yazdırılıyor. Kuralı burada
  değiştirirsen orada da değiştir.
* Değişikliği Gemini'ye gitmeden görmek için: `python -m horoscope --dry-run`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta

from .signs import HOUSE_THEMES, SIGNS, Sign, house_of
from .sky import Sky, day_label

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

SYSTEM = """\
Sen, Astarot adlı Türkçe tarot ve burç uygulaması için günlük burç yorumları \
yazan deneyimli bir astrologsun. Okurların, her gün uygulamayı açıp kendi \
burcunun yorumunu okuyan, astrolojiye meraklı ama uzman olmayan insanlar.

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

_WEEKDAYS = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")


def build_prompt(sky: Sky) -> str:
    return (
        _context(sky)
        + "\n\nOn iki burcun hepsini Koç'tan Balık'a sırayla yaz. \"burc\" "
        "alanına burcun adını yukarıdaki yazımla koy."
    )


def build_repair_prompt(
    sky: Sky,
    readings: dict[str, dict[str, str]],
    issues: dict[str, list[str]],
) -> str:
    """Kuralları çiğneyen burçları, sorunlarıyla birlikte yeniden yazdıran istem."""
    lines = [
        _context(sky),
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


def response_schema(signs: Sequence[Sign] = SIGNS) -> dict:
    entry = {
        "type": "object",
        "properties": {
            "burc": {"type": "string", "enum": [s.name for s in signs]},
            **{
                field: {"type": "string", "description": spec}
                for field, spec in FIELD_SPECS.items()
            },
        },
        "required": ["burc", *FIELDS],
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


def _transit(body: str, sign: Sign, at: Sign, ingress) -> str:
    house = house_of(sign, at)
    text = f"{body} {house}. evinde ({HOUSE_THEMES[house - 1]})"
    if ingress:
        later = house_of(sign, ingress.sign)
        clock = (ingress.at + timedelta(seconds=30)).strftime("%H:%M")
        text += f", saat {clock} itibarıyla {later}. evine geçiyor ({HOUSE_THEMES[later - 1]})"
    return text
