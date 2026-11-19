import numpy as np
import pandas as pd

from src.config.feature_engineering import TemporalFeatureEngineer


def build_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [0.0, 3600.0, 82800.0, 172792.0],
            "Amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_temporal_engineer_adds_cyclic_features_without_dropping_columns():
    engineer = TemporalFeatureEngineer()
    transformed = engineer.fit_transform(build_sample())

    for column in ["Time", "Amount", "hour", "hour_sin", "hour_cos"]:
        assert column in transformed.columns
    assert len(transformed) == 4


def test_temporal_engineer_hour_is_within_day_range():
    transformed = TemporalFeatureEngineer().fit_transform(build_sample())

    assert transformed["hour"].between(0.0, 23.0).all()
    assert transformed["hour_sin"].between(-1.0, 1.0).all()
    assert transformed["hour_cos"].between(-1.0, 1.0).all()


def test_temporal_engineer_cyclic_identity_holds():
    transformed = TemporalFeatureEngineer().fit_transform(build_sample())

    magnitude = np.square(transformed["hour_sin"]) + np.square(transformed["hour_cos"])
    assert np.allclose(magnitude, 1.0)


def test_temporal_engineer_is_row_wise_and_deterministic():
    engineer = TemporalFeatureEngineer()
    sample = build_sample()
    first = engineer.fit_transform(sample)
    second = engineer.transform(sample)

    pd.testing.assert_frame_equal(first, second)


def test_temporal_engineer_feature_names_out():
    engineer = TemporalFeatureEngineer()
    sample = build_sample()
    engineer.fit(sample)

    names = list(engineer.get_feature_names_out(sample.columns))
    assert names == [*sample.columns, "hour", "hour_sin", "hour_cos"]
