"""12 burç — uygulamadaki `ZodiacSign` ile aynı sıra, ad ve anahtarlar.

`slug` yayınlanan JSON'da burcun anahtarı; uygulama `ZodiacSign.slug` ile
eşliyor. Yönetici gezegenler de uygulamadaki gibi klasik astrolojiden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sign:
    slug: str
    name: str
    element: str
    modality: str
    ruler: str


SIGNS: tuple[Sign, ...] = (
    Sign("aries", "Koç", "Ateş", "Öncü", "Mars"),
    Sign("taurus", "Boğa", "Toprak", "Sabit", "Venüs"),
    Sign("gemini", "İkizler", "Hava", "Değişken", "Merkür"),
    Sign("cancer", "Yengeç", "Su", "Öncü", "Ay"),
    Sign("leo", "Aslan", "Ateş", "Sabit", "Güneş"),
    Sign("virgo", "Başak", "Toprak", "Değişken", "Merkür"),
    Sign("libra", "Terazi", "Hava", "Öncü", "Venüs"),
    Sign("scorpio", "Akrep", "Su", "Sabit", "Mars"),
    Sign("sagittarius", "Yay", "Ateş", "Değişken", "Jüpiter"),
    Sign("capricorn", "Oğlak", "Toprak", "Öncü", "Satürn"),
    Sign("aquarius", "Kova", "Hava", "Sabit", "Satürn"),
    Sign("pisces", "Balık", "Su", "Değişken", "Jüpiter"),
)

# Güneş burcu evleri: her burç kendi burcunu 1. ev sayar. Günlük burç
# yorumlarının klasik yöntemi — doğum saati bilinmeden de her burca kendi
# vurgusunu veriyor.
HOUSE_THEMES: tuple[str, ...] = (
    "kimlik, beden ve kendini ortaya koyuş",
    "para, kaynaklar ve öz değer",
    "iletişim, yakın çevre ve kısa yolculuklar",
    "ev, aile ve kökler",
    "aşk, yaratıcılık ve keyif",
    "günlük düzen, iş yükü ve sağlık alışkanlıkları",
    "ilişkiler, ortaklıklar ve karşındaki insanlar",
    "ortak kaynaklar, yakınlık ve dönüşüm",
    "inançlar, uzak yerler, öğrenme ve anlam arayışı",
    "kariyer, hedefler ve toplumdaki yer",
    "arkadaşlar, topluluk ve gelecek planları",
    "iç dünya, dinlenme ve sezgiler",
)


def sign_of(longitude: float) -> Sign:
    """Tropikal ekliptik boylamdan (derece) burç."""
    return SIGNS[int((longitude % 360) // 30) % len(SIGNS)]


def house_of(sign: Sign, other: Sign) -> int:
    """`sign` 1. ev sayıldığında `other` burcunun ev numarası (1–12)."""
    return (SIGNS.index(other) - SIGNS.index(sign)) % len(SIGNS) + 1


def find_sign(name: object) -> Sign | None:
    """"Koç", "KOÇ", "Koç Burcu" ya da "aries" yazımlarından burcu bulur."""
    key = _key(name)
    for sign in SIGNS:
        if key in (_key(sign.name), sign.slug):
            return sign
    return None


def _key(text: object) -> str:
    # Python'un lower()'ı Türkçe bilmiyor: "İ" → "i̇" (noktalı), "I" → "i".
    value = str(text or "").replace("İ", "i").replace("I", "ı").lower()
    return value.replace("burcu", "").strip()
