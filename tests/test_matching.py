"""
Tests du module de Matching inter-documentaire.
"""

from src.matching.matcher import CrossDocumentMatcher
from src.validation.schemas import (
    DocumentExtraction, ExtractedField, DocumentTypeEnum, MatchingStatusEnum
)


def test_cross_matching_cin_match():
    cin_doc = DocumentExtraction(
        document_id="cin_passeport",
        filename="cin.jpg",
        document_type=DocumentTypeEnum.CIN,
        fields={
            "cin_passeport": ExtractedField(value="05826391", source_document="cin.jpg"),
            "nom_prenom_acheteur": ExtractedField(value="Abdelkarim Habria", source_document="cin.jpg")
        }
    )
    contrat_doc = DocumentExtraction(
        document_id="contrat",
        filename="contrat.jpg",
        document_type=DocumentTypeEnum.CONTRAT,
        fields={
            "cin_passeport": ExtractedField(value="05826391", source_document="contrat.jpg"),
            "nom_prenom_acheteur": ExtractedField(value="Abdelkarim Habria", source_document="contrat.jpg"),
            "numero_chassis": ExtractedField(value="9MWX05047", source_document="contrat.jpg"),
            "cylindree": ExtractedField(value=49, source_document="contrat.jpg")
        }
    )

    results = CrossDocumentMatcher.compare_documents([cin_doc, contrat_doc])
    cin_matches = [r for r in results if r.field == "cin_passeport"]
    assert len(cin_matches) == 1
    assert cin_matches[0].status == MatchingStatusEnum.MATCH


def test_cross_matching_mismatch_triggers():
    cin_doc = DocumentExtraction(
        document_id="cin_passeport",
        filename="cin.jpg",
        document_type=DocumentTypeEnum.CIN,
        fields={
            "cin_passeport": ExtractedField(value="05826391", source_document="cin.jpg")
        }
    )
    contrat_doc = DocumentExtraction(
        document_id="contrat",
        filename="contrat.jpg",
        document_type=DocumentTypeEnum.CONTRAT,
        fields={
            "cin_passeport": ExtractedField(value="09999999", source_document="contrat.jpg")
        }
    )

    results = CrossDocumentMatcher.compare_documents([cin_doc, contrat_doc])
    cin_matches = [r for r in results if r.field == "cin_passeport"]
    assert len(cin_matches) == 1
    assert cin_matches[0].status == MatchingStatusEnum.MISMATCH


# --- Une ressemblance approximative ne doit jamais produire MATCH ---

def _doc(name, doc_type, **fields):
    return DocumentExtraction(
        document_id=name, filename=name, document_type=doc_type,
        fields={k: ExtractedField(value=v, source_document=name) for k, v in fields.items()},
    )


def _status(results, field):
    return [r.status for r in results if r.field == field]


def test_same_name_in_other_order_is_match():
    a = _doc("a", DocumentTypeEnum.CIN, nom_prenom_acheteur="Habria Abdelkarim")
    b = _doc("b", DocumentTypeEnum.CONTRAT, nom_prenom_acheteur="abdelkarim HABRIA")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "nom_prenom_acheteur") == [MatchingStatusEnum.MATCH]


def test_close_but_different_name_is_review_not_match():
    a = _doc("a", DocumentTypeEnum.CIN, nom="Habria")
    b = _doc("b", DocumentTypeEnum.CONTRAT, nom="Habriya")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "nom") == [MatchingStatusEnum.REVIEW]


def test_partial_name_is_review_not_match():
    a = _doc("a", DocumentTypeEnum.CIN, nom_prenom_acheteur="Abdelkarim Habria")
    b = _doc("b", DocumentTypeEnum.CONTRAT, nom_prenom_acheteur="Habria")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "nom_prenom_acheteur") == [MatchingStatusEnum.REVIEW]


def test_different_names_are_mismatch():
    a = _doc("a", DocumentTypeEnum.CIN, nom="Habria")
    b = _doc("b", DocumentTypeEnum.CONTRAT, nom="Slimani")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "nom") == [MatchingStatusEnum.MISMATCH]


def test_truncated_chassis_start_is_match():
    a = _doc("a", DocumentTypeEnum.CONTRAT, numero_chassis="LBAMPAB3421")
    b = _doc("b", DocumentTypeEnum.QUITTANCE, numero_chassis="LBAMPAB342")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "numero_chassis") == [MatchingStatusEnum.MATCH]


def test_identical_chassis_ignores_separators_and_case():
    a = _doc("a", DocumentTypeEnum.CONTRAT, numero_chassis="9MWX05047")
    b = _doc("b", DocumentTypeEnum.QUITTANCE, numero_chassis="9mwx-05047")
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "numero_chassis") == [MatchingStatusEnum.MATCH]


def test_displacement_conflict_is_mismatch():
    a = _doc("a", DocumentTypeEnum.CONTRAT, cylindree=49)
    b = _doc("b", DocumentTypeEnum.QUITTANCE, cylindree=125)
    assert _status(CrossDocumentMatcher.compare_documents([a, b]), "cylindree") == [MatchingStatusEnum.MISMATCH]


def test_carte_grise_is_compared_with_quittance():
    cg = _doc("cg", DocumentTypeEnum.CARTE_GRISE, numero_chassis="ABC12345", cylindree=50, date_mec="01/03/2019")
    q = _doc("q", DocumentTypeEnum.QUITTANCE, numero_chassis="ABC12345", cylindree=50)
    res = CrossDocumentMatcher.compare_documents([cg, q])
    assert _status(res, "numero_chassis") == [MatchingStatusEnum.MATCH]
    assert _status(res, "cylindree") == [MatchingStatusEnum.MATCH]


def test_sale_dates_compared_only_between_sale_documents():
    contrat = _doc("c", DocumentTypeEnum.CONTRAT, date_vente="11/02/2021")
    facture = _doc("f", DocumentTypeEnum.FACTURE, date_vente="11/02/2021")
    quittance = _doc("q", DocumentTypeEnum.QUITTANCE, date_vente="12/01/2026")   # date de la quittance : autre événement
    res = CrossDocumentMatcher.compare_documents([contrat, facture, quittance])
    assert _status(res, "date_vente") == [MatchingStatusEnum.MATCH]


def test_conflicting_sale_dates_are_mismatch():
    contrat = _doc("c", DocumentTypeEnum.CONTRAT, date_vente="11/02/2021")
    facture = _doc("f", DocumentTypeEnum.FACTURE, date_vente="12/02/2021")
    assert _status(CrossDocumentMatcher.compare_documents([contrat, facture]), "date_vente") == [MatchingStatusEnum.MISMATCH]


def test_missing_value_is_never_compared():
    a = _doc("a", DocumentTypeEnum.CIN, cin_passeport="05826391")
    b = _doc("b", DocumentTypeEnum.CONTRAT, cin_passeport=None)
    assert CrossDocumentMatcher.compare_documents([a, b]) == []
