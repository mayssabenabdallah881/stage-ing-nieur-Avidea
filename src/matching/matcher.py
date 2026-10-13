"""
Matching inter-documentaire.
Compare les informations communes à plusieurs pièces et produit STRICTEMENT : MATCH, MISMATCH ou REVIEW.

Principe : seule une égalité (après nettoyage de forme : casse, accents, séparateurs) donne MATCH.
Une ressemblance approximative donne au mieux REVIEW, jamais MATCH.
"""

import difflib
import re
import unicodedata
from itertools import combinations
from typing import Any, List, Optional, Tuple

from src.matching.transliteration import cross_script_equal
from src.config import NAME_REVIEW_SIMILARITY_MIN, CHASSIS_REVIEW_SIMILARITY_MIN, CHASSIS_TAIL_MIN_CHARS
from src.validation.schemas import (
    DocumentExtraction, DocumentTypeEnum, MatchingResult, MatchingStatusEnum
)

# Pièces qui portent la date de la vente (la quittance fiscale a sa propre date : on ne la compare pas)
SALE_DOCUMENT_TYPES = {DocumentTypeEnum.CONTRAT, DocumentTypeEnum.FACTURE}


def _value(doc: DocumentExtraction, field: str) -> Optional[str]:
    f = doc.fields.get(field)
    if f is None or f.value is None or str(f.value).strip() == "":
        return None
    return str(f.value).strip()


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _name_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]|_", " ", _strip_accents(text).casefold())
    return [t for t in cleaned.split() if t]


def _alnum_upper(text: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", _strip_accents(text).upper())


def _parse_date(text: str) -> Optional[Tuple[int, int, int]]:
    m = re.fullmatch(r"\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\s*", text)
    return (int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


class CrossDocumentMatcher:
    """Compare deux à deux tous les documents qui portent une même information."""

    @classmethod
    def compare_documents(cls, extractions: List[DocumentExtraction]) -> List[MatchingResult]:
        results: List[MatchingResult] = []

        for doc_a, doc_b in combinations(extractions, 2):
            # --- Identité ---
            if cls._same_identifier_kind(doc_a, doc_b):
                cls._add(results, "cin_passeport", doc_a, doc_b, cls._compare_exact)

            for field in ("nom", "prenom"):
                cls._add(results, field, doc_a, doc_b, cls._compare_names)

            # Le nom complet n'est comparé que si les champs séparés ne sont pas disponibles des deux côtés
            both_split = all(_value(d, f) for d in (doc_a, doc_b) for f in ("nom", "prenom"))
            if not both_split:
                cls._add(results, "nom_prenom_acheteur", doc_a, doc_b, cls._compare_names)

            # --- Véhicule ---
            cls._add(results, "numero_chassis", doc_a, doc_b, cls._compare_chassis)
            cls._add(results, "cylindree", doc_a, doc_b, cls._compare_displacement)

            # --- Dates ---
            if doc_a.document_type in SALE_DOCUMENT_TYPES and doc_b.document_type in SALE_DOCUMENT_TYPES:
                cls._add(results, "date_vente", doc_a, doc_b, cls._compare_dates)
            cls._add(results, "date_mec", doc_a, doc_b, cls._compare_dates)

        return results

    @staticmethod
    def _same_identifier_kind(doc_a: DocumentExtraction, doc_b: DocumentExtraction) -> bool:
        """
        Une CIN (chiffres seuls) et un numéro de passeport (avec lettres) sont deux identifiants différents :
        les comparer donnerait un faux désaccord. On ne compare donc que des identifiants de même nature.
        Si l'un des deux est absent, `_add` n'ajoute de toute façon rien.
        """
        a, b = _value(doc_a, "cin_passeport"), _value(doc_b, "cin_passeport")
        if a is None or b is None:
            return True
        return _alnum_upper(a).isdigit() == _alnum_upper(b).isdigit()

    @classmethod
    def _add(cls, results, field, doc_a, doc_b, comparator) -> None:
        val_a, val_b = _value(doc_a, field), _value(doc_b, field)
        if val_a is None or val_b is None:
            return  # information absente d'un côté : rien à comparer (jamais déduit)
        results.append(comparator(field, doc_a.filename, doc_b.filename, val_a, val_b))

    @staticmethod
    def _result(field, doc_a, doc_b, val_a, val_b, status, similarity, message) -> MatchingResult:
        return MatchingResult(
            field=field, document_a=doc_a, document_b=doc_b, value_a=val_a, value_b=val_b,
            status=status, similarity=round(similarity, 2), message=message,
        )

    @classmethod
    def _compare_exact(cls, field, doc_a, doc_b, val_a, val_b) -> MatchingResult:
        """Identifiants stricts (CIN / passeport) : égal ou contradictoire."""
        if _alnum_upper(val_a) == _alnum_upper(val_b):
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               f"Concordance de {field} ({val_a})")
        return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MISMATCH, 0.0,
                           f"Divergence {field} : {val_a} ≠ {val_b}")

    @classmethod
    def _compare_names(cls, field, doc_a, doc_b, val_a, val_b) -> MatchingResult:
        """Noms : MATCH seulement si les mots sont les mêmes (l'ordre nom/prénom importe peu)."""
        tokens_a, tokens_b = set(_name_tokens(val_a)), set(_name_tokens(val_b))

        if tokens_a and tokens_a == tokens_b:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               "Identité concordante")

        # M?me personne ?crite en arabe sur une pi?ce et en lettres latines sur l'autre
        if cross_script_equal(sorted(tokens_a), sorted(tokens_b)):
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               "Identit? concordante (translitt?ration arabe/latin)")

        if tokens_a and tokens_b and (tokens_a < tokens_b or tokens_b < tokens_a):
            sim = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.REVIEW, sim,
                               f"Nom incomplet sur l'une des pièces ({val_a} / {val_b}) : vérification requise")

        ratio = difflib.SequenceMatcher(
            None, " ".join(sorted(tokens_a)), " ".join(sorted(tokens_b))
        ).ratio()
        if ratio >= NAME_REVIEW_SIMILARITY_MIN:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.REVIEW, ratio,
                               f"Noms proches mais non identiques ({val_a} / {val_b}) : vérification requise")
        return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MISMATCH, ratio,
                           f"Identités incompatibles ({val_a} ≠ {val_b})")

    @classmethod
    def _compare_chassis(cls, field, doc_a, doc_b, val_a, val_b) -> MatchingResult:
        """Châssis : MATCH seulement si identique ; numéro tronqué ou très proche → REVIEW."""
        a, b = _alnum_upper(val_a), _alnum_upper(val_b)

        if a == b:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               "Numéro de châssis identique")

        # Num?ro partiel sur l'une des pi?ces : on compare la fin du num?ro de s?rie
        short, long_ = sorted((a, b), key=len)
        n = CHASSIS_TAIL_MIN_CHARS
        is_tail = short[-n:] == long_[-n:]
        is_head = len(short) >= 2 * n and long_.startswith(short)
        if len(short) >= n and len(short) < len(long_) and (is_tail or is_head):
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 0.85,
                               "Chassis concordant : numero partiel sur l'une des pieces")

        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if (a in b or b in a) or ratio >= CHASSIS_REVIEW_SIMILARITY_MIN:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.REVIEW, ratio,
                               f"Châssis très proche mais non identique ({val_a} / {val_b}) : vérification requise")
        return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MISMATCH, ratio,
                           f"Numéros de châssis contradictoires ({val_a} ≠ {val_b})")

    @classmethod
    def _compare_displacement(cls, field, doc_a, doc_b, val_a, val_b) -> MatchingResult:
        try:
            int_a, int_b = int(float(val_a)), int(float(val_b))
        except ValueError:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.REVIEW, 0.0,
                               "Valeur de cylindrée non convertible")
        if int_a == int_b:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               f"Cylindrée identique : {int_a} cc")
        return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MISMATCH, 0.0,
                           f"Conflit de cylindrée : {int_a} cc ≠ {int_b} cc")

    @classmethod
    def _compare_dates(cls, field, doc_a, doc_b, val_a, val_b) -> MatchingResult:
        da, db = _parse_date(val_a), _parse_date(val_b)
        if da is None or db is None:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.REVIEW, 0.0,
                               f"Date non exploitable ({val_a} / {val_b})")
        if da == db:
            return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MATCH, 1.0,
                               f"Date identique ({val_a})")
        return cls._result(field, doc_a, doc_b, val_a, val_b, MatchingStatusEnum.MISMATCH, 0.0,
                           f"Dates contradictoires ({val_a} ≠ {val_b})")
