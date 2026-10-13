"""
Codes normalisés du rapport (chapitre 8) : codes de rejet REJ-xx et anomalies FRA-xx.

Les anomalies FRA-xx sont des SIGNAUX DE RISQUE à faire contrôler par un humain, jamais des preuves de fraude.
"""

from typing import Dict

REJECTION_CODES: Dict[str, str] = {
    "REJ-01": "DISPLACEMENT_ABOVE_LIMIT",
    "REJ-02": "DATE_CONDITION_NOT_MET",
    "REJ-03": "BUSINESS_INELIGIBILITY",
}

ANOMALY_CODES: Dict[str, str] = {
    "FRA-01": "CHASSIS_MISMATCH",
    "FRA-02": "IDENTITY_MISMATCH",
    "FRA-03": "CIN_MISMATCH",
    "FRA-04": "DISPLACEMENT_CONFLICT",
    "FRA-05": "DATE_INCONSISTENCY",
    "FRA-06": "POSSIBLE_DOCUMENT_TAMPERING",
    "FRA-07": "DUPLICATE_DOCUMENT",
    "FRA-08": "UNKNOWN_DOCUMENT",
    "FRA-09": "SUSPICIOUS_FIELD",
}
