import pandas as pd
import pytest

from src.config.prepare_data import validate_input_dataset, validate_prepared_dataset
from src.config.train_model import evaluate_model_acceptance


def test_validate_input_dataset_rejects_missing_required_column():
    df = pd.DataFrame(
        {
            "Time": [0, 1, 2, 3],
            "Class": [0, 1, 0, 1],
        }
    )
    quality_config = {
        "minimum_rows": 4,
        "allow_nulls": False,
        "required_columns": ["Time", "Amount", "Class"],
        "target_column": "Class",
        "target_allowed_values": [0, 1],
    }

    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        validate_input_dataset(df, quality_config)


def test_validate_prepared_dataset_accepts_raw_features():
    """A saida do prepare_data agora mantem features BRUTAS (anti-leakage).

    A transformacao (Yeo-Johnson/MinMax) passou a ocorrer dentro do Pipeline de
    treino, ajustada por fold. Portanto a validacao nao pode mais exigir Time em
    [0, 1]; ela apenas reporta os valores observados.
    """
    df = pd.DataFrame(
        {
            "Time": [10.0, 500.0, 172792.0],
            "Amount": [1.0, 100.5, 25691.16],
            "Class": [0, 1, 0],
        }
    )
    quality_config = {
        "minimum_rows": 3,
        "allow_nulls": False,
        "required_columns": ["Time", "Amount", "Class"],
        "target_column": "Class",
        "target_allowed_values": [0, 1],
    }
    preprocess_config = {
        "time_scaler": "minmax",
        "amount_transform": "yeo-johnson",
    }

    report = validate_prepared_dataset(df, quality_config, preprocess_config)

    assert report["row_count"] == 3
    assert report["null_count"] == 0
    assert report["time_min"] == 10.0
    assert report["time_max"] == 172792.0
    assert report["features_are_raw"] is True
    assert report["preprocessing_stage"] == "training_pipeline"


def test_evaluate_model_acceptance_passes_when_metrics_meet_gate():
    metrics = {
        "cross_validation_mean_metrics": {
            "precision": 0.9230,
            "recall": 0.8216,
        },
        "threshold_calibration": {
            "selected": {
                "metrics": {
                    "precision": 0.9222,
                    "recall": 0.8191,
                },
                "business_alignment": {
                    "weighted_gap": 0.0099,
                },
            }
        },
    }
    acceptance_config = {
        "min_cv_precision": 0.92,
        "min_cv_recall": 0.81,
        "min_calibrated_precision": 0.92,
        "min_calibrated_recall": 0.81,
        "max_business_weighted_gap": 0.02,
    }

    acceptance = evaluate_model_acceptance(metrics, acceptance_config)

    assert acceptance["passed"] is True
    assert all(acceptance["checks"].values())
