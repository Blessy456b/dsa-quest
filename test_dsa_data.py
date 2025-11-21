# tests/test_dsa_data.py

from dsa_data import STRIVER_SHEET, TOTAL_PROBLEMS


def test_total_problems_matches_structure():
    counted = sum(len(step["problems"]) for step in STRIVER_SHEET.values())
    assert counted == TOTAL_PROBLEMS


def test_problem_ids_unique():
    ids = []
    for step in STRIVER_SHEET.values():
        for p in step["problems"]:
            ids.append(p["id"])
    assert len(ids) == len(set(ids))
