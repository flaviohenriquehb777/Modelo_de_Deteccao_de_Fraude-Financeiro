import json

import yaml

from src.config.paths import PROJECT_ROOT
from src.config.train_model import evaluate_model_acceptance, summarize_aspirational_target


def load_metrics() -> dict:
    with (PROJECT_ROOT / "metrics.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_metrics(precision: float, recall: float, weighted_gap: float = 0.005) -> dict:
    return {
        "cross_validation_mean_metrics": {"precision": precision, "recall": recall},
        "threshold_calibration": {
            "selected": {
                "metrics": {"precision": precision, "recall": recall},
                "business_alignment": {"weighted_gap": weighted_gap},
            }
        },
    }


def test_gate_blocks_when_recall_below_minimum():
    """O gate deve REPROVAR se a meta de negocio nao for atingida (RuntimeError)."""
    acceptance_config = load_params()["train"]["acceptance"]
    below_target = build_metrics(precision=0.96, recall=0.79)

    acceptance = evaluate_model_acceptance(below_target, acceptance_config)

    assert acceptance["passed"] is False
    assert acceptance["checks"]["min_calibrated_recall"] is False


def test_gate_passes_at_business_target():
    acceptance_config = load_params()["train"]["acceptance"]
    at_target = build_metrics(precision=0.9608, recall=0.8028)

    acceptance = evaluate_model_acceptance(at_target, acceptance_config)

    assert acceptance["passed"] is True
    assert all(acceptance["checks"].values())


def test_aspirational_target_is_reported_and_non_blocking():
    metrics = load_metrics()
    aspirational = metrics["aspirational_target"]

    assert aspirational is not None
    assert aspirational["blocking"] is False
    assert aspirational["target_precision"] == 0.96
    assert aspirational["target_recall"] == 0.82
    assert aspirational["simultaneously_achievable"] is False
    # A meta aspiracional NAO pode constar entre os criterios do gate bloqueante.
    assert not any("aspirational" in key for key in metrics["acceptance"]["criteria"])


def test_summarize_aspirational_target_uses_honest_trade_off():
    evaluation_config = load_params()["train"]["evaluation"]
    trade_off = [
        {"recall_floor": 0.80, "threshold": 0.566, "precision": 0.9611, "recall": 0.8028},
        {"recall_floor": 0.82, "threshold": 0.186, "precision": 0.9140, "recall": 0.8211},
    ]

    summary = summarize_aspirational_target(trade_off, evaluation_config)

    assert summary["target_precision"] == 0.96
    assert summary["target_recall"] == 0.82
    assert summary["best_feasible"]["precision"] == 0.9140
    assert summary["best_feasible"]["recall"] == 0.8211
    assert round(summary["precision_gap"], 4) == round(0.96 - 0.9140, 4)
    assert summary["simultaneously_achievable"] is False


def test_summarize_aspirational_target_returns_none_without_config():
    assert summarize_aspirational_target([], {"target_precision": 0.96}) is None
