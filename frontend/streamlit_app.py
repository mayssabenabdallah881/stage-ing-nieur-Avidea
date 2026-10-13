"""
Interface Streamlit de l'Agentic AI de souscription d'assurance motos (rapport, chapitre 11). Palette AVIDEA : #087F7B, #075E5B, #F5F7F8.

Vues : Dashboard | Nouveau dossier | Agent Workspace | Résultat | Dossiers | Historique.
L'interface ne contient AUCUNE logique métier : elle appelle le backend API (frontend/api_client.py).

    python run.py api        # terminal 1 : backend
    python run.py app        # terminal 2 : interface   (ou : streamlit run frontend/streamlit_app.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import api_client as api

st.set_page_config(page_title="Agentic AI de souscription d'assurance motos", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F5F7F8; color: #17202A; }
    .agent-header { background: linear-gradient(90deg, #087F7B 0%, #075E5B 100%); color: white; padding: 14px 24px;
        border-radius: 8px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .agent-title { font-size: 20px; font-weight: 700; }
    .badge-status { padding: 5px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; display: inline-block; }
    .badge-accepted { background-color: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-review { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .metric-card { background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; }
    .metric-val { font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-lbl { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #6B7280; }
    .step-card { background: white; padding: 12px 16px; margin-bottom: 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

DECISION_LABEL = {"ACCEPTED": ("Accepté", "badge-accepted"), "REJECTED": ("Rejeté", "badge-rejected"),
                  "REVIEW_REQUIRED": ("Revue requise", "badge-review"),
                  "DOCUMENT_MANQUANT": ("Document manquant", "badge-review")}
SCORE_LABELS = [
    ("extraction_score", "Extraction Score (confiance déclarée par le VLM)"),
    ("completeness_score", "Completeness Score"),
    ("validity_score", "Validity Score"),
    ("identity_match_score", "Identity Match Score"),
    ("chassis_match_score", "Chassis Match Score"),
    ("displacement_consistency", "Displacement Consistency"),
    ("date_consistency", "Date Consistency"),
    ("global_data_confidence", "Global Data Confidence"),
]
PIPELINE = [("Documents reçus", "RECEIVED"), ("Quality Check", "QUALITY_CHECK"), ("Classification", "CLASSIFICATION"),
            ("Extraction", "EXTRACTION_VLM"), ("Validation Pydantic", "VALIDATION"), ("Reconstruction", "RECONSTRUCTION"),
            ("Normalisation", "NORMALIZATION"), ("Matching", "MATCHING"), ("Qualité des données et anomalies", "DATA_QUALITY"),
            ("Validation métier", "BUSINESS_RULES"), ("Décision", "DECISION"), ("Persistance SQLite", "PERSISTENCE")]


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="agent-header"><div class="agent-title">{title}</div><div>{subtitle}</div></div>',
                unsafe_allow_html=True)


def card(label: str, value, color: str = "#17202A") -> str:
    return (f'<div class="metric-card"><div class="metric-lbl" style="color:{color}">{label}</div>'
            f'<div class="metric-val" style="color:{color}">{value}</div></div>')


def pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f} %"


def badge(decision: str) -> str:
    label, css = DECISION_LABEL.get(decision, (decision, "badge-review"))
    return f'<span class="badge-status {css}">{label}</span>'


def call(fn, *args, **kwargs):
    """Appel API ; en cas d'échec, message clair puis arrêt de la vue."""
    try:
        return fn(*args, **kwargs)
    except api.ApiUnavailable as e:
        st.error(str(e))
    except api.ApiError as e:
        st.error(f"Erreur de l'API : {e}")
    st.stop()


with st.sidebar:
    st.markdown('<h2 style="color:#087F7B;margin:0">Agentic AI</h2>'
                '<p style="color:#6B7280;font-size:12px">Souscription d\'assurance motos • VLM Agent</p>', unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🏠 Dashboard", "📁 Nouveau dossier", "🤖 Agent Workspace",
                                   "📊 Résultat", "📂 Dossiers", "🕒 Historique"])
    st.markdown("---")
    st.caption("Vision-Language Model • Agentic AI • Décision déterministe et explicable • SQLite")

health = call(api.health)
if health["offline_mock"]:
    st.warning("MODE TEST HORS-LIGNE : extractions écrites à la main, non issues du modèle. Aucune performance n'est mesurée.")
elif not health["vlm_available"]:
    st.error("Aucune clé GEMINI_API_KEY côté API : les documents ne pourront pas être lus (dossiers en REVIEW_REQUIRED).")

# --------------------------------------------------------------------------- Dashboard
if menu == "🏠 Dashboard":
    header("🏠 Tableau de bord", f"Modèle : {health['model']}")
    s = call(api.stats)
    cols = st.columns(7)
    cols[0].markdown(card("Dossiers analysés", s["total_dossiers"]), unsafe_allow_html=True)
    cols[1].markdown(card("Acceptés", s["accepted_count"], "#168A5B"), unsafe_allow_html=True)
    cols[2].markdown(card("Rejetés", s["rejected_count"], "#C0392B"), unsafe_allow_html=True)
    cols[3].markdown(card("Revue requise", s["review_count"], "#D97706"), unsafe_allow_html=True)
    cols[6].markdown(card("Document manquant", s["missing_document_count"], "#92400E"), unsafe_allow_html=True)
    cols[4].markdown(card("Anomalies", s["anomalies_total"], "#92400E"), unsafe_allow_html=True)
    cols[5].markdown(card("Taux de fallback", f"{s['fallback_rate']} %", "#087F7B"), unsafe_allow_html=True)

    st.markdown("### Indicateurs de qualité")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Qualité technique moyenne des images", pct(s["avg_quality"]))
    q2.metric("Complétude moyenne", pct(s["avg_completeness"]))
    q3.metric("Confiance globale des données (indicateur non calibré)", pct(s["avg_global_data_confidence"]))
    q4.metric("Dossiers avec anomalie", s["dossiers_with_anomalies"])

    st.markdown("### Évaluation sur vérité terrain")
    if s["evaluated_dossiers"]:
        e1, e2, e3 = st.columns(3)
        e1.metric("Dossiers évalués", s["evaluated_dossiers"])
        e2.metric("Exact match moyen (champs)", pct(s["mean_exact_match"]))
        e3.metric("Décisions conformes à la référence", pct(s["decision_accuracy"]))
        st.caption("Corpus restreint : ces valeurs décrivent les dossiers annotés, pas l'ensemble des dossiers d'assurance.")
    else:
        st.caption(f"{s['annotated_dossiers']} dossiers annotés. Aucune évaluation disponible : elle exige une vérité terrain "
                   "et de vraies extractions du modèle (voir `python run.py evaluate`).")

    st.markdown("### Derniers dossiers")
    rows = call(api.list_dossiers)
    if rows:
        st.dataframe(pd.DataFrame(rows)[["dossier_id", "created_at", "decision", "rejection_type", "docs_count", "anomalies_count", "reason"]],
                     width="stretch", hide_index=True)
    else:
        st.info("Aucun dossier analysé pour le moment.")

# --------------------------------------------------------------------------- Nouveau dossier
elif menu == "📁 Nouveau dossier":
    header("📁 Nouveau dossier", "Chargement de plusieurs documents")
    uploaded = st.file_uploader("Déposez les pièces du dossier (CIN, passeport, contrat, facture, quittance, carte grise, permis) :",
                                accept_multiple_files=True, type=["jpeg", "jpg", "png", "pdf"])
    sample_list = call(api.samples)
    choice = st.selectbox("…ou lancez un dossier d'exemple :", [""] + [f"{d['dossier_id']} — {d['client']} ({d['documents']} pièces)" for d in sample_list])

    if st.button("🚀 Lancer l'analyse", type="primary"):
        if uploaded:
            payload = [(f.name, f.getvalue()) for f in uploaded]
            with st.spinner("L'agent analyse les documents (qualité, classification, extraction, matching, règles)…"):
                res = call(api.analyze_uploads, payload)
        elif choice:
            with st.spinner("L'agent analyse le dossier d'exemple…"):
                res = call(api.analyze_sample, choice.split(" — ")[0])
        else:
            st.warning("Sélectionnez un dossier d'exemple ou déposez des documents.")
            st.stop()
        st.session_state["current_dossier_id"] = res["dossier_id"]
        label = DECISION_LABEL.get(res["decision"], (res["decision"],))[0]
        st.success(f"Analyse terminée : {res['dossier_id']} — {label}")
        st.write(res["reason"])
        if res["offline_mock_used"]:
            st.warning("Résultat obtenu en MODE TEST HORS-LIGNE : valeurs écrites à la main.")
        if res["persistence_error"]:
            st.error(f"Résultat non enregistré dans SQLite : {res['persistence_error']}")
        st.info("Consultez « Agent Workspace » (étapes) ou « Résultat » (détail).")

# --------------------------------------------------------------------------- sélection du dossier courant
def pick_dossier():
    rows = call(api.list_dossiers)
    if not rows:
        st.info("Aucun dossier enregistré.")
        st.stop()
    ids = [r["dossier_id"] for r in rows]
    current = st.session_state.get("current_dossier_id")
    idx = ids.index(current) if current in ids else 0
    chosen = st.selectbox("Dossier", ids, index=idx)
    st.session_state["current_dossier_id"] = chosen
    return call(api.get_dossier, chosen)


# --------------------------------------------------------------------------- Agent Workspace
if menu == "🤖 Agent Workspace":
    header("🤖 Agent Workspace", "Étapes du traitement et re-tentatives")
    d = pick_dossier()
    traces = d["traces"]
    st.markdown("#### Pipeline")
    for label, step in PIPELINE:
        statuses = [t["status"] for t in traces if t["step"] == step and t["status"] != "RUNNING"]
        if not statuses:
            mark = "[ — ]" if step not in ("PREPROCESSING",) else "[ — ]"
        elif "FAILED" in statuses:
            mark = "[ÉCHEC]"
        elif any(x in ("WARNING", "FALLBACK") for x in statuses):
            mark = "[ATTENTION]"
        else:
            mark = "[OK]"
        st.code(f"{mark} {label}", language=None)

    st.markdown("#### Journal détaillé")
    for i, t in enumerate(traces):
        color = "#087F7B" if t["status"] in ("SUCCESS", "RUNNING") else ("#D97706" if t["status"] in ("WARNING", "FALLBACK") else "#C0392B")
        st.markdown(f'<div class="step-card" style="border-left:4px solid {color}"><b>{i+1}. {t["step"]}</b> '
                    f'<span style="color:#6B7280;float:right">{t["timestamp"]} • {t["duration_ms"]} ms</span>'
                    f'<div style="font-size:14px">{t["message"]}</div></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- Résultat
elif menu == "📊 Résultat":
    header("📊 Résultat de l'analyse", "Décision explicable")
    d = pick_dossier()
    dec = d["decision"]
    details = d.get("decision_details") or {}
    st.markdown(f'<div class="metric-card"><div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h3 style="margin:0">{d["dossier_id"]}</h3>{badge(dec)}</div><hr>'
                f'<p><b>Justification :</b> {d["reason"]}</p>'
                + (f'<p><b>Type de rejet :</b> {d["rejection_type"]} ({", ".join(details["rejection_codes"])})</p>' if d.get("rejection_type") else "")
                + "</div>", unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4, t5, t6 = st.tabs(["📋 Données extraites", "🗂️ Dossier reconstruit", "🔗 Matching", "⚙️ Règles appliquées",
                                       "📈 Scores", "⚠️ Anomalies"])
    with t1:
        if d["fields"]:
            st.dataframe(pd.DataFrame(d["fields"])[["filename", "field_name", "value", "raw_value", "confidence", "status"]],
                         width="stretch", hide_index=True)
        st.caption("Une valeur absente ou illisible reste vide (null) : elle n'est jamais estimée.")
    with t2:
        rec = d.get("reconstruction")
        if rec:
            c1, c2, c3 = st.columns(3)
            c1.write("**Identité**"); c1.write(rec["identite"] or "—")
            c2.write("**Véhicule**"); c2.write(rec["vehicule"] or "—")
            c3.write("**Autres**"); c3.write(rec["autres"] or "—")
            if rec["non_identifies"]:
                st.warning(f"Pièces non identifiées ou illisibles : {', '.join(rec['non_identifies'])}")
            if rec["pieces_manquantes"]:
                st.info("Catégories sans pièce : " + ", ".join(rec["pieces_manquantes"]))
            st.write("**Valeurs consolidées**")
            st.json({k: {"valeur": v, "source": rec["consolidated_source"].get(k)} for k, v in rec["consolidated"].items()})
    with t3:
        if d["matching"]:
            st.dataframe(pd.DataFrame(d["matching"])[["field", "document_a", "document_b", "value_a", "value_b", "status", "message"]],
                         width="stretch", hide_index=True)
        else:
            st.info("Aucune comparaison inter-documents possible.")
    with t4:
        st.dataframe(pd.DataFrame(d["rules"])[["rule_name", "status", "code", "rejection_type", "message"]], width="stretch", hide_index=True)
        if details.get("explanation"):
            with st.expander("Justification détaillée"):
                for line in details["explanation"]:
                    st.write("• " + line)
    with t5:
        sc = d.get("scores")
        if sc:
            st.dataframe(pd.DataFrame([{"Indicateur": label, "Valeur": pct(sc.get(key))} for key, label in SCORE_LABELS]),
                         width="stretch", hide_index=True)
            st.write(f"Champs manquants : **{sc['missing_field_count']}** / {sc['fields_total']}")
            st.caption("Indicateurs interprétables du prototype, NON calibrés : ce ne sont pas des probabilités. "
                       "« n/a » = non calculable (aucune comparaison possible).")
    with t6:
        if d["anomalies"]:
            st.dataframe(pd.DataFrame(d["anomalies"])[["code", "name", "severity", "field", "message"]],
                         width="stretch", hide_index=True)
            st.caption("Signaux de risque à faire contrôler, pas des preuves de fraude.")
        else:
            st.success("Aucune anomalie détectée.")

# --------------------------------------------------------------------------- Dossiers
elif menu == "📂 Dossiers":
    header("📂 Dossiers enregistrés", "Base SQLite")
    rows = call(api.list_dossiers)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("La base ne contient aucun dossier.")

# --------------------------------------------------------------------------- Historique
elif menu == "🕒 Historique":
    header("🕒 Historique et traçabilité", "Journal d'audit de chaque dossier")
    rows = call(api.list_dossiers)
    if not rows:
        st.info("Aucun dossier enregistré.")
    else:
        st.dataframe(pd.DataFrame(rows)[["created_at", "dossier_id", "decision", "anomalies_count", "fallback_triggered"]],
                     width="stretch", hide_index=True)
        chosen = st.selectbox("Consulter la trace d'un dossier", [r["dossier_id"] for r in rows])
        st.dataframe(pd.DataFrame(call(api.get_dossier, chosen)["traces"]), width="stretch", hide_index=True)
