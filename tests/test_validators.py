"""Tests des validateurs (étape VALIDATION) et du bilan de qualité des données (étape DATA_QUALITY)."""

from src.validation.schemas import (
    DocumentExtraction, ExtractedField, DocumentTypeEnum, MatchingResult, MatchingStatusEnum,
)
from src.validation.validators import validate_extraction, assess_data_quality


def _ext(name="d.jpg", doc_type=DocumentTypeEnum.CONTRAT, **values):
    return DocumentExtraction(
        document_id=name, filename=name, document_type=doc_type,
        fields={k: ExtractedField(value=v, source_document=name) for k, v in values.items()},
    )


def test_valid_extraction_has_no_issue():
    ext = _ext(cin_passeport="05826391", cylindree=49, numero_chassis="9MWX05047", date_vente="11/02/2021", nom="Habria")
    assert validate_extraction(ext) == []


def test_invalid_values_are_flagged_but_kept():
    ext = _ext(cin_passeport="123", cylindree=5, numero_chassis="AB", date_vente="31/02/2021", date_mec="01/01/2999", nom="1234")
    issues = validate_extraction(ext)
    assert {i["field"] for i in issues} == {"cin_passeport", "cylindree", "numero_chassis", "date_vente", "date_mec", "nom"}
    assert ext.fields["cin_passeport"].value == "123"
    assert all(ext.fields[f].status == "AMBIGUOUS" for f in ("cin_passeport", "cylindree", "date_vente"))


def test_null_values_are_not_issues():
    assert validate_extraction(_ext(cin_passeport=None, cylindree=None)) == []


def test_data_quality_counts_are_real():
    a = _ext("a.jpg", DocumentTypeEnum.CIN, cin_passeport="05826391", nom_prenom_acheteur="A B")
    b = _ext("b.jpg", DocumentTypeEnum.CONTRAT, numero_chassis="9MWX05047", cylindree=49)
    matching = [
        MatchingResult(field="cin_passeport", document_a="a", document_b="b", status=MatchingStatusEnum.MATCH),
        MatchingResult(field="nom", document_a="a", document_b="b", status=MatchingStatusEnum.REVIEW),
    ]
    dq = assess_data_quality([a, b], matching)
    assert dq["fields_present"] == 4 and dq["fields_total"] == 8
    assert dq["critical_fields_missing"] == []
    assert dq["matching"] == {"MATCH": 1, "MISMATCH": 0, "REVIEW": 1}
    assert dq["status"] == "ATTENTION"          # un REVIEW reste à vérifier


def test_data_quality_reports_missing_critical_fields():
    dq = assess_data_quality([_ext(cin_passeport="05826391")], [])
    assert set(dq["critical_fields_missing"]) == {"nom_prenom_acheteur", "numero_chassis", "cylindree"}
