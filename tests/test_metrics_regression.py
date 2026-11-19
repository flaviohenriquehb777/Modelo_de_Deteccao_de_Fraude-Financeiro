import json

import yaml

from src.config.paths import PROJECT_ROOT


def load_metrics() -> dict:
    with (PROJECT_ROOT / "metrics.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_acceptance_gate_passed():
    metrics = load_metrics()
    assert metrics["acceptance"]["passed"] is True


def test_metrics_meet_acceptance_thresholds():
    params = load_params()
    acceptance = params["train"]["acceptance"]
    metrics = load_metrics()

    cv_metrics = metrics["evaluation"]["cross_validation_mean_metrics"]
    threshold_calibration = metrics["evaluation"]["threshold_calibration"]["selected"]
    calibrated_metrics = threshold_calibration["metrics"]
    weighted_gap = threshold_calibration["business_alignment"]["weighted_gap"]

    assert cv_metrics["precision"] >= acceptance["min_cv_precision"]
    assert cv_metrics["recall"] >= acceptance["min_cv_recall"]
    assert calibrated_metrics["precision"] >= acceptance["min_calibrated_precision"]
    assert calibrated_metrics["recall"] >= acceptance["min_calibrated_recall"]
    assert weighted_gap <= acceptance["max_business_weighted_gap"]


def test_threshold_selection_respects_recall_floor():
    params = load_params()
    evaluation = params["train"]["evaluation"]
    metrics = load_metrics()

    calibration = metrics["evaluation"]["threshold_calibration"]
    selected = calibration["selected"]

    assert calibration["selection_rule"] == "maximize_precision_subject_to_recall"
    assert 0.0 <= selected["threshold"] <= 1.0
    assert selected["metrics"]["recall"] >= evaluation["target_recall"]
    assert calibration["search_space"]["candidates"] >= 2

