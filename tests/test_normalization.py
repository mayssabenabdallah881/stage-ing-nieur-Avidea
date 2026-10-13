"""
Tests unitaires de normalisation des champs (CIN, cylindrée, dates, châssis).
"""

import pytest
from src.validation.normalization import (
    normalize_cin, normalize_chassis, normalize_date,
    normalize_displacement, normalize_name
)


def test_normalize_displacement():
    assert normalize_displacement("49 cm3")[0] == 49
    assert normalize_displacement("125 CC")[0] == 125
    assert normalize_displacement("49cm³")[0] == 49
    assert normalize_displacement("150")[0] == 150
    assert normalize_displacement(None)[0] is None


def test_normalize_cin():
    assert normalize_cin("05826391")[0] == "05826391"
    assert normalize_cin("5826391")[0] == "05826391"  # 7 chiffres -> pad 0
    assert normalize_cin("06039883")[0] == "06039883"
    assert normalize_cin(None)[0] is None


def test_normalize_chassis():
    assert normalize_chassis("9MWX-05047")[0] == "9MWX05047"
    assert normalize_chassis("LB AMP AB 342")[0] == "LBAMPAB342"
    assert normalize_chassis(None)[0] is None


def test_normalize_date():
    assert normalize_date("29/03/1977")[0] == "29/03/1977"
    assert normalize_date("1977-03-29")[0] == "29/03/1977"
    assert normalize_date("29 mars 1977")[0] == "29/03/1977"
    assert normalize_date(None)[0] is None
