"""
Module de normalisation déterministe des champs extraits des documents.
Convertit les formats bruts multilingues (arabe, français) en formes canoniques comparables.
"""

import re
from typing import Optional, Tuple, Union
from datetime import datetime


def normalize_displacement(raw_val: Optional[Union[str, int, float]]) -> Tuple[Optional[int], Optional[str]]:
    """
    Normalise la cylindrée (ex: '49 cm3', '125 CC', '49cm³', '125.0') en entier (ex: 49, 125).
    Retourne (valeur_normalisée, valeur_brute).
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    if not raw_str:
        return None, None

    # Extraction des chiffres
    match = re.search(r"(\d{2,4})", raw_str)
    if match:
        try:
            val = int(match.group(1))
            return val, raw_str
        except ValueError:
            pass

    return None, raw_str


def normalize_cin(raw_val: Optional[Union[str, int]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalise le numéro de CIN tunisienne (8 chiffres avec zéro initial si nécessaire).
    Ex: '5826391' -> '05826391'.
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    digits = re.sub(r"\D", "", raw_str)

    if not digits:
        return None, raw_str

    if len(digits) == 7:
        # Ajout du zéro initial standard en Tunisie
        digits = "0" + digits

    if len(digits) == 8:
        return digits, raw_str

    return digits if digits else None, raw_str


def normalize_cin_passeport(raw_val: Optional[Union[str, int]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalise le champ `cin_passeport`.
    - valeur composée uniquement de chiffres (et séparateurs) : CIN tunisienne -> normalize_cin (8 chiffres) ;
    - valeur contenant des lettres : numéro de passeport -> majuscules, sans espaces ni séparateurs.
    Le type de pièce (CIN ou passeport) est contrôlé ensuite par la validation, jamais deviné ici.
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    if not raw_str or raw_str.lower() in ("null", "none"):
        return None, None

    if re.search(r"[A-Za-z]", raw_str):
        clean_val = re.sub(r"[^0-9A-Za-z]", "", raw_str).upper()
        return (clean_val or None), raw_str

    return normalize_cin(raw_str)


def normalize_chassis(raw_val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalise le numéro de châssis / VIN / numéro d'immatriculation.
    Supprime les espaces, tirets et met en majuscules.
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    clean_val = re.sub(r"[\s\-_.:/]", "", raw_str).upper()

    if not clean_val or clean_val.lower() in ["null", "none", "inconnu", "sans"]:
        return None, raw_str

    return clean_val, raw_str


def normalize_date(raw_val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalise une date sous le format canonique 'JJ/MM/AAAA'.
    Gère les séparateurs '/', '-', '.' et les mois en arabe/français.
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    if not raw_str or raw_str.lower() in ["null", "none", ""]:
        return None, None

    # Mois en arabe et en français
    month_mapping = {
        "janvier": "01", "janv": "01", "جانفي": "01", "يناير": "01",
        "fevrier": "02", "février": "02", "fevr": "02", "فيفري": "02", "فبراير": "02",
        "mars": "03", "مارس": "03",
        "avril": "04", "avr": "04", "افريل": "04", "أفريل": "04", "ابريل": "04",
        "mai": "05", "ماي": "05", "مايو": "05",
        "juin": "06", "جوان": "06", "يونيو": "06",
        "juillet": "07", "juil": "07", "جويلية": "07", "يوليو": "07",
        "aout": "08", "août": "08", "اوت": "08", "أوت": "08", "اغسطس": "08",
        "septembre": "09", "sept": "09", "سبتمبر": "09",
        "octobre": "10", "oct": "10", "اكتوبر": "10", "أكتوبر": "10",
        "novembre": "11", "nov": "11", "نوفمبر": "11",
        "decembre": "12", "décembre": "12", "dec": "12", "ديسمبر": "12",
    }

    # Recherche directe format numérique DD/MM/YYYY ou YYYY-MM-DD
    # Cas 1: YYYY-MM-DD ou YYYY/MM/DD
    match_iso = re.search(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", raw_str)
    if match_iso:
        y, m, d = match_iso.group(1), match_iso.group(2).zfill(2), match_iso.group(3).zfill(2)
        return f"{d}/{m}/{y}", raw_str

    # Cas 2: DD/MM/YYYY ou DD-MM-YYYY
    match_std = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", raw_str)
    if match_std:
        d, m, y = match_std.group(1).zfill(2), match_std.group(2).zfill(2), match_std.group(3)
        return f"{d}/{m}/{y}", raw_str

    # Cas 3: Texte avec mois littéral (ex: '29 مارس 1977', '12 septembre 1989')
    for m_name, m_num in month_mapping.items():
        if m_name in raw_str.lower():
            match_txt = re.search(r"(\d{1,2})\s+" + re.escape(m_name) + r"\s+(\d{4})", raw_str, re.IGNORECASE)
            if match_txt:
                d = match_txt.group(1).zfill(2)
                y = match_txt.group(2)
                return f"{d}/{m_num}/{y}", raw_str

    return raw_str, raw_str


def normalize_name(raw_val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Nettoie et normalise les chaînes de nom/prénom.
    Supprime les titres superflus ('M.', 'Mme', 'السيد', 'المشتري').
    """
    if raw_val is None:
        return None, None

    raw_str = str(raw_val).strip()
    clean_val = raw_str

    prefixes = [
        r"^السيد\s*:", r"^السيد\s+", r"^السيدة\s+", r"^المشتري\s*:", r"^البائع\s*:",
        r"^M\.\s*", r"^Mme\s*", r"^DoitMr\s*", r"^Doit\s*"
    ]
    for p in prefixes:
        clean_val = re.sub(p, "", clean_val, flags=re.IGNORECASE).strip()

    clean_val = re.sub(r"\s+", " ", clean_val).strip()

    if not clean_val or clean_val.lower() in ["null", "none"]:
        return None, raw_str

    return clean_val, raw_str
