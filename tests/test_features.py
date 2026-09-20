import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, RAW_COLUMNS, make_features


def one_row(**overrides):
    row = {c: 0.0 for c in RAW_COLUMNS}
    row.update(overrides)
    return pd.DataFrame([row])


def test_output_columns_match_feature_list():
    assert list(make_features(one_row()).columns) == FEATURE_COLUMNS


def test_missing_column_raises_clear_error():
    with pytest.raises(ValueError, match="Amount"):
        make_features(one_row().drop(columns=["Amount"]))


def test_time_of_day_wraps_after_24_hours():
    a = make_features(one_row(Time=3_600))
    b = make_features(one_row(Time=3_600 + 86_400))
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_amount_is_log_scaled():
    f = make_features(one_row(Amount=np.e - 1))
    assert f["log_amount"].iloc[0] == pytest.approx(1.0)
