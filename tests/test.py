import pandas as pd

from src.preprocessing import resolve_id_column


def test_resolve_id_column_uppercase():
    df = pd.DataFrame({"ID": [1, 2], "target": [0, 1]})
    assert resolve_id_column(df) == "ID"


def test_resolve_id_column_lowercase():
    df = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    assert resolve_id_column(df) == "id"
