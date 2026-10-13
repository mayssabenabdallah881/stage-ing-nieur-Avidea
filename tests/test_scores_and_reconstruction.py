"""Scores interprétables (chapitre 9) et reconstruction du dossier (chapitre 7)."""

from src.agent.reconstruction import reconstruct_dossier, consolidate_fields
from src.validation.schemas import (
    DocumentExtraction, ExtractedField, DocumentTypeEnum as T, ExtractionStatusEnum,
    MatchingResult, MatchingStatusEnum as M,
)
from src.validation.scoring import compute_scores


def doc(name, doc_type, conf=0.9, status="VALID", **values):
    return DocumentExtraction(
        document_id=name, filename=name, document_type=doc_type,
        fields={k: ExtractedField(value=v, confidence=conf, status=status, source_document=name) for k, v in values.items()})


def match(field, status):
    return MatchingResult(field=field, document_a="a", document_b="b", status=status)


def test_scores_are_counts_and_ratios():
    docs = [doc("a", T.CIN, cin_passeport="05826391", nom_prenom_acheteur="A B"),
            doc("b", T.CONTRAT, numero_chassis="X12345", cylindree=49)]
    s = compute_scores(docs, [match("cin_passeport", M.MATCH), match("nom", M.REVIEW), match("numero_chassis", M.MATCH)])
    assert s.missing_field_count == 4 and s.fields_total == 8
    assert s.completeness_score == 0.5
    assert s.extraction_score == 0.9 and s.validity_score == 1.0
    assert s.identity_match_score == 0.5 and s.identity_match == 0          # un REVIEW n'est pas un MATCH
    assert s.chassis_match_score == 1.0 and s.chassis_match == 1


def test_not_comparable_scores_are_none_not_invented():
    s = compute_scores([doc("a", T.CIN, cin_passeport="05826391")], [])
    assert s.identity_match is None and s.chassis_match is None and s.date_consistent is None
    assert s.displacement_consistency is None
    assert "chassis_match_score" not in s.components_used


def test_empty_dossier_only_has_the_measured_completeness():
    s = compute_scores([doc("a", T.UNKNOWN)], [])
    assert s.missing_field_count == 8 and s.completeness_score == 0.0
    assert s.extraction_score is None and s.validity_score is None
    assert s.components_used == ["completeness_score"] and s.global_data_confidence == 0.0   # seul fait mesuré : 0 champ


def test_validity_score_penalises_ambiguous_values():
    docs = [doc("a", T.CONTRAT, cylindree=5, status="AMBIGUOUS")]
    assert compute_scores(docs, []).validity_score == 0.0


def test_global_confidence_is_weighted_mean_of_available_scores():
    docs = [doc("a", T.CIN, conf=1.0, cin_passeport="05826391")]
    s = compute_scores(docs, [match("cin_passeport", M.MATCH)])
    assert 0.0 <= s.global_data_confidence <= 1.0
    assert set(s.components_used) == {"extraction_score", "completeness_score", "validity_score", "identity_match_score"}


# --- Reconstruction ---

def test_documents_are_grouped_as_in_the_report():
    docs = [doc("cin", T.CIN), doc("pp", T.PASSPORT), doc("f", T.FACTURE), doc("c", T.CONTRAT), doc("cg", T.CARTE_GRISE),
            doc("q", T.QUITTANCE), doc("p", T.PERMIS), doc("u", T.UNKNOWN)]
    r = reconstruct_dossier(docs)
    assert r.identite == ["cin", "pp"] and r.vehicule == ["f", "c"] and r.autres == ["cg", "q", "p"]   # carte grise : « autres »
    assert r.non_identifies == ["u"] and r.pieces_manquantes == []


def test_missing_categories_are_reported_but_unreadable_docs_are_not_typed():
    bad = doc("bad", T.CIN)
    bad.extraction_status = ExtractionStatusEnum.UNREADABLE
    r = reconstruct_dossier([bad, doc("q", T.QUITTANCE)])
    assert r.non_identifies == ["bad"]
    assert len(r.pieces_manquantes) == 2 and "CIN ou PASSPORT" in r.pieces_manquantes[0]


def test_consolidation_prefers_the_most_relevant_document_and_never_fills_gaps():
    docs = [doc("q", T.QUITTANCE, cylindree=50, numero_chassis="Q1"),
            doc("cg", T.CARTE_GRISE, cylindree=49, date_mec=None)]
    values = consolidate_fields(docs)
    assert values["cylindree"] == 49 and values["numero_chassis"] == "Q1"   # carte grise d'abord ; sinon quittance
    assert "date_mec" not in values                                          # absent partout : reste absent
