"""
Tests unitaires des règles métier (rapport, chapitre 8 et annexe B).

Seuils : <= 49 cc condition satisfaite ; > 125 cc REJECTED (REJ-01) ;
49 < cc <= 125 : condition sur la date de mise en circulation (date_mec), jamais remplacée par la date de vente.
"""

from src.business.rules import BusinessRulesEngine
from src.business.decision import UnderwritingDecisionEngine
from src.config import load_business_rules
from src.validation.schemas import (
    Anomaly, BusinessRuleResult, BusinessRuleStatusEnum,
    DocumentExtraction, ExtractedField, DocumentTypeEnum,
    MatchingResult, MatchingStatusEnum, DecisionEnum
)


def make_dummy_extraction(displacement, date_mec=None, date_vente="01/01/2023",
                          cin_passeport="05826391", chassis="9MWX05047"):
    return DocumentExtraction(
        document_id="doc1",
        filename="contrat.jpg",
        document_type=DocumentTypeEnum.CONTRAT,
        fields={
            "cylindree": ExtractedField(value=displacement, source_document="contrat.jpg"),
            "date_vente": ExtractedField(value=date_vente, source_document="contrat.jpg"),
            "date_mec": ExtractedField(value=date_mec, source_document="contrat.jpg"),
            "cin_passeport": ExtractedField(value=cin_passeport, source_document="contrat.jpg"),
            "numero_chassis": ExtractedField(value=chassis, source_document="contrat.jpg"),
        }
    )


def complete(exts):
    """Ajoute les pièces obligatoires absentes (documents vides) pour isoler la règle testée."""
    types = {e.document_type for e in exts}
    out = list(exts)
    if not types & {DocumentTypeEnum.CIN, DocumentTypeEnum.PASSPORT}:
        out.append(DocumentExtraction(document_id="id", filename="cin.jpg", document_type=DocumentTypeEnum.CIN))
    if not types & {DocumentTypeEnum.CONTRAT, DocumentTypeEnum.FACTURE}:
        out.append(DocumentExtraction(document_id="v", filename="vente.jpg", document_type=DocumentTypeEnum.CONTRAT))
    return out


def decide(ext, matching=None, **kw):
    rules = BusinessRulesEngine.evaluate_dossier(complete([ext]), matching or [])
    return rules, UnderwritingDecisionEngine.compute_decision(rules, **kw)


def mismatch(field, a="X1", b="Y2"):
    return MatchingResult(field=field, document_a="a", document_b="b", value_a=a, value_b=b,
                          status=MatchingStatusEnum.MISMATCH, similarity=0.0, message=f"{field} contradictoire")


# --- Cylindrée ---

def test_small_engine_approved():
    _, d = decide(make_dummy_extraction(49))
    assert d.decision == DecisionEnum.ACCEPTED and d.rejection_codes == []


def test_oversized_engine_rejected_with_rej_01():
    _, d = decide(make_dummy_extraction(150))
    assert d.decision == DecisionEnum.REJECTED
    assert d.rejection_codes == ["REJ-01"]
    assert "150 cc supérieure au seuil maximal autorisé (125 cc)" in d.reason
    assert "DISPLACEMENT_ABOVE_LIMIT" in d.reason


def test_125cc_is_not_above_the_limit():
    _, d = decide(make_dummy_extraction(125, date_mec="01/03/2019"))
    assert d.decision == DecisionEnum.ACCEPTED


# --- Condition de date (49 < cc <= 125) : date_mec explicite, jamais substituée ---

def test_mid_displacement_with_recent_date_mec_rejected_with_rej_02():
    _, d = decide(make_dummy_extraction(100, date_mec="15/06/2025"))
    assert d.decision == DecisionEnum.REJECTED and d.rejection_codes == ["REJ-02"]


def test_mid_displacement_with_old_date_mec_approved():
    _, d = decide(make_dummy_extraction(100, date_mec="15/06/2019"))
    assert d.decision == DecisionEnum.ACCEPTED


def test_missing_date_mec_is_never_replaced_by_date_vente():
    # date_vente postérieure à 2025 : si elle était substituée, le dossier serait rejeté.
    rules, d = decide(make_dummy_extraction(100, date_mec=None, date_vente="15/06/2025"))
    date_rule = next(r for r in rules if r.rule_name == "DATE_CONDITION")
    assert d.decision == DecisionEnum.REVIEW_REQUIRED
    assert date_rule.status == BusinessRuleStatusEnum.REVIEW_REQUIRED and date_rule.missing_data
    assert date_rule.evidence["date_vente_observee"] == "15/06/2025"    # montrée au gestionnaire, non utilisée
    assert d.rejection_codes == []


def test_unparseable_date_mec_requires_review_not_approval():
    _, d = decide(make_dummy_extraction(100, date_mec="2019"))
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_date_condition_not_evaluated_for_small_engine():
    rules, _ = decide(make_dummy_extraction(49))
    assert all(r.rule_name != "DATE_CONDITION" for r in rules)


# --- Données manquantes, contradictions et ordre de priorité (annexe B) ---

def test_missing_mandatory_field_review():
    ext = DocumentExtraction(
        document_id="doc1", filename="scan.jpg", document_type=DocumentTypeEnum.CIN,
        fields={
            "cin_passeport": ExtractedField(value="05826391", source_document="scan.jpg"),
            "numero_chassis": ExtractedField(value=None, source_document="scan.jpg"),
        }
    )
    _, d = decide(ext)
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_missing_critical_data_has_priority_over_rejection():
    # annexe B : « if missing_critical_data: REVIEW_REQUIRED » passe avant « elif displacement > 125 »
    _, d = decide(make_dummy_extraction(150, cin_passeport=None))
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_missing_displacement_requires_review():
    _, d = decide(make_dummy_extraction(None))
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_cross_document_contradiction_requires_review_not_rejection():
    _, d = decide(make_dummy_extraction(49), [mismatch("numero_chassis")])
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_contradictory_displacement_never_causes_rejection():
    # 49 cc sur une pièce, 150 cc sur l'autre : la valeur est incertaine, on ne rejette pas sur l'une des deux
    rules, d = decide(make_dummy_extraction(150), [mismatch("cylindree", "49", "150")])
    assert d.decision == DecisionEnum.REVIEW_REQUIRED and d.rejection_codes == []
    assert next(r for r in rules if r.rule_name == "DISPLACEMENT_CHECK").status == BusinessRuleStatusEnum.REVIEW_REQUIRED


def test_ambiguous_displacement_field_requires_review():
    ext = make_dummy_extraction(150)
    ext.fields["cylindree"].status = "AMBIGUOUS"
    _, d = decide(ext)
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_anomalies_and_review_reasons_turn_approval_into_review_only():
    ok = make_dummy_extraction(49)
    anomaly = Anomaly(code="FRA-08", name="UNKNOWN_DOCUMENT", severity="MEDIUM", message="type non identifié")
    _, d = decide(ok, anomalies=[anomaly])
    assert d.decision == DecisionEnum.REVIEW_REQUIRED
    _, d = decide(ok, review_reasons=["Contradiction entre tentatives"])
    assert d.decision == DecisionEnum.REVIEW_REQUIRED and "Contradiction" in d.reason
    # ils ne peuvent jamais produire un rejet : un rejet n'existe que par une règle FAILED
    _, d = decide(make_dummy_extraction(150), anomalies=[anomaly], review_reasons=["x"])
    assert d.decision == DecisionEnum.REJECTED


def test_failed_rule_without_specific_code_gives_rej_03():
    rule = BusinessRuleResult(rule_name="OTHER", status=BusinessRuleStatusEnum.FAILED, message="Inéligibilité métier")
    d = UnderwritingDecisionEngine.compute_decision([rule])
    assert d.decision == DecisionEnum.REJECTED and d.rejection_codes == ["REJ-03"]


def test_explanation_lists_every_rule():
    rules, d = decide(make_dummy_extraction(49))
    assert len(d.explanation) == len(rules)
    assert all(line.startswith("[PASSED]") for line in d.explanation)


# --- Configuration externalisée ---

def test_thresholds_are_read_from_yaml(tmp_path):
    f = tmp_path / "rules.yaml"
    f.write_text("max_allowed_displacement: 200\nsmall_engine_threshold: 50\n", encoding="utf-8")
    cfg = load_business_rules(f)
    assert cfg["max_allowed_displacement"] == 200 and cfg["small_engine_threshold"] == 50
    assert cfg["restriction_year"] == 2025 and cfg["restriction_date_field"] == "date_mec"   # défauts conservés


def test_default_thresholds_match_the_report():
    cfg = load_business_rules()
    assert cfg["max_allowed_displacement"] == 125 and cfg["small_engine_threshold"] == 49


# --- Pièces obligatoires : (CIN ou PASSPORT) ET (CONTRAT ou FACTURE), sinon DOCUMENT_MANQUANT ---

def typed(t, **values):
    return DocumentExtraction(document_id=t.value, filename=f"{t.value}.jpg", document_type=t,
                              fields={k: ExtractedField(value=v, source_document="x") for k, v in values.items()})


FULL = dict(cin_passeport="05826391", numero_chassis="9MWX05047", cylindree=49)


def decide_docs(*docs):
    rules = BusinessRulesEngine.evaluate_dossier(list(docs), [])
    return rules, UnderwritingDecisionEngine.compute_decision(rules)


def test_cin_and_contrat_are_enough():
    _, d = decide_docs(typed(DocumentTypeEnum.CIN), typed(DocumentTypeEnum.CONTRAT, **FULL))
    assert d.decision == DecisionEnum.ACCEPTED


def test_passport_and_facture_are_accepted_alternatives():
    _, d = decide_docs(typed(DocumentTypeEnum.PASSPORT), typed(DocumentTypeEnum.FACTURE, **FULL))
    assert d.decision == DecisionEnum.ACCEPTED


def test_missing_identity_document_gives_document_manquant():
    rules, d = decide_docs(typed(DocumentTypeEnum.CONTRAT, **FULL))
    assert d.decision == DecisionEnum.DOCUMENT_MANQUANT and "IDENTITE" in d.reason
    assert next(r for r in rules if r.rule_name == "REQUIRED_DOCUMENTS").status == BusinessRuleStatusEnum.DOCUMENT_MANQUANT


def test_missing_sale_document_gives_document_manquant():
    _, d = decide_docs(typed(DocumentTypeEnum.CIN, **FULL))
    assert d.decision == DecisionEnum.DOCUMENT_MANQUANT and "VEHICULE" in d.reason


def test_quittance_carte_grise_permis_do_not_replace_required_documents():
    _, d = decide_docs(typed(DocumentTypeEnum.CIN), typed(DocumentTypeEnum.CARTE_GRISE, **FULL),
                       typed(DocumentTypeEnum.QUITTANCE), typed(DocumentTypeEnum.PERMIS))
    assert d.decision == DecisionEnum.DOCUMENT_MANQUANT and "VEHICULE" in d.reason


def test_both_missing_are_listed():
    _, d = decide_docs(typed(DocumentTypeEnum.QUITTANCE))
    assert "IDENTITE" in d.reason and "VEHICULE" in d.reason


def test_missing_document_has_priority_over_rejection_and_review():
    _, d = decide_docs(typed(DocumentTypeEnum.CONTRAT, cin_passeport="05826391", numero_chassis="X12345", cylindree=150))
    assert d.decision == DecisionEnum.DOCUMENT_MANQUANT and d.rejection_codes == []


def test_unreadable_document_is_not_concluded_as_absent():
    # une pièce non identifiée peut être celle qui manque : revue humaine, pas DOCUMENT_MANQUANT
    unreadable = DocumentExtraction(document_id="u", filename="u.jpg", document_type=DocumentTypeEnum.UNKNOWN)
    _, d = decide_docs(typed(DocumentTypeEnum.CONTRAT, **FULL), unreadable)
    assert d.decision == DecisionEnum.REVIEW_REQUIRED


def test_required_documents_come_from_the_yaml(tmp_path):
    f = tmp_path / "r.yaml"
    f.write_text("required_documents:\n  identity: [CIN]\n  vehicle: [CONTRAT, FACTURE, CARTE_GRISE]\n", encoding="utf-8")
    cfg = load_business_rules(f)["required_documents"]
    assert cfg == {"identity": ["CIN"], "vehicle": ["CONTRAT", "FACTURE", "CARTE_GRISE"], "payment": []}
    f.write_text("required_documents:\n  identity: []\n", encoding="utf-8")
    assert load_business_rules(f)["required_documents"]["identity"] == []          # liste vide : exigence désactivée


def test_unknown_document_type_in_yaml_is_rejected(tmp_path):
    import pytest
    f = tmp_path / "r.yaml"
    f.write_text("required_documents:\n  identity: [CARTE_VERTE]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_business_rules(f)


def test_shipped_yaml_requires_identity_sale_document_and_quittance():
    from src.config import CONFIG_DIR
    cfg = load_business_rules(CONFIG_DIR / "business_rules.yaml")["required_documents"]
    assert cfg == {"identity": ["CIN", "PASSPORT"], "vehicle": ["CONTRAT", "FACTURE"], "payment": ["QUITTANCE"]}


# --- rejection_type (configurable) et décision ACCEPTED ---

def test_rejection_type_above_125():
    _, d = decide(make_dummy_extraction(150))
    assert d.decision == DecisionEnum.REJECTED and d.rejection_type == "PUISSANCE_SUP_125"


def test_rejection_type_for_mid_displacement_after_2025():
    _, d = decide(make_dummy_extraction(100, date_mec="02/01/2025"))
    assert d.decision == DecisionEnum.REJECTED and d.rejection_type == "PUISSANCE_SUP_49_APRES_2025"


def test_boundaries_exactly_as_specified():
    assert decide(make_dummy_extraction(49))[1].decision == DecisionEnum.ACCEPTED                     # <= 49
    assert decide(make_dummy_extraction(50, date_mec="01/01/2025"))[1].decision == DecisionEnum.ACCEPTED   # date_mec = 01/01/2025 : pas « > »
    assert decide(make_dummy_extraction(50, date_mec="02/01/2025"))[1].decision == DecisionEnum.REJECTED  # date_mec > 01/01/2025
    assert decide(make_dummy_extraction(125, date_mec="01/03/2019"))[1].decision == DecisionEnum.ACCEPTED # <= 125
    assert decide(make_dummy_extraction(126))[1].rejection_type == "PUISSANCE_SUP_125"                 # > 125


def test_accepted_and_review_have_no_rejection_type():
    assert decide(make_dummy_extraction(49))[1].rejection_type is None
    assert decide(make_dummy_extraction(100))[1].rejection_type is None      # date_mec absente : revue


def test_decision_enum_uses_accepted():
    assert [e.value for e in DecisionEnum] == ["ACCEPTED", "REJECTED", "REVIEW_REQUIRED", "DOCUMENT_MANQUANT"]


def test_rejection_types_come_from_the_yaml(tmp_path):
    f = tmp_path / "r.yaml"
    f.write_text("rejection_types:\n  displacement_above_limit: AUTRE_LIBELLE\n", encoding="utf-8")
    cfg = load_business_rules(f)["rejection_types"]
    assert cfg["displacement_above_limit"] == "AUTRE_LIBELLE" and cfg["date_condition_not_met"] == "PUISSANCE_SUP_49_APRES_2025"


def test_quittance_required_when_configured(monkeypatch):
    """Avec payment: [QUITTANCE] (regles livrees), un dossier sans quittance est DOCUMENT_MANQUANT."""
    import src.agent.reconstruction as rc
    monkeypatch.setattr(rc, "PAYMENT_TYPES", (DocumentTypeEnum.QUITTANCE,))
    monkeypatch.setattr(rc, "REQUIRED_PAYMENT_TYPES", ("QUITTANCE",))
    _, d = decide_docs(typed(DocumentTypeEnum.CIN), typed(DocumentTypeEnum.CONTRAT, **FULL))
    assert d.decision == DecisionEnum.DOCUMENT_MANQUANT and "PAIEMENT" in d.reason
    _, d = decide_docs(typed(DocumentTypeEnum.CIN), typed(DocumentTypeEnum.CONTRAT, **FULL), typed(DocumentTypeEnum.QUITTANCE))
    assert d.decision == DecisionEnum.ACCEPTED
