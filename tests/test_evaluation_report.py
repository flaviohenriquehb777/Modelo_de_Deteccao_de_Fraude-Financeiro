import json

import pytest
import yaml

from src.config.paths import EVALUATION_REPORT_MD_PATH, EVALUATION_REPORT_PATH, PROJECT_ROOT


@pytest.fixture(scope="module")
def report() -> dict:
    with EVALUATION_REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def evaluation_config() -> dict:
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["train"]["evaluation"]


def test_evaluation_report_artifacts_exist():
    assert EVALUATION_REPORT_PATH.exists(), "Relatorio JSON de avaliacao ausente"
    assert EVALUATION_REPORT_MD_PATH.exists(), "Relatorio Markdown de avaliacao ausente"


def test_fold_metrics_cover_every_fold(report, evaluation_config):
    fold_metrics = report["cross_validation"]["fold_metrics"]
    assert len(fold_metrics) == evaluation_config["cv_folds"]
    for fold in fold_metrics:
        assert set(fold) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_summary_reports_mean_and_std_for_every_metric(report):
    summary = report["cross_validation"]["summary"]
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert metric_name in summary
        assert set(summary[metric_name]) == {"mean", "std"}
        assert summary[metric_name]["std"] >= 0.0


def test_confusion_matrix_is_consistent_with_dataset(report):
    matrix = report["out_of_fold"]["confusion_matrix"]
    total = report["dataset"]["row_count"]
    positives = report["dataset"]["positive_count"]

    confusion_total = sum(
        (
            matrix["true_negative"],
            matrix["false_positive"],
            matrix["false_negative"],
            matrix["true_positive"],
        )
    )
    assert confusion_total == total
    assert matrix["true_positive"] + matrix["false_negative"] == positives


def test_probability_metrics_are_bounded(report):
    probability_metrics = report["out_of_fold"]["probability_metrics"]
    assert 0.0 <= probability_metrics["roc_auc"] <= 1.0
    assert 0.0 <= probability_metrics["pr_auc"] <= 1.0


def test_seed_robustness_covers_configured_seeds(report, evaluation_config):
    robustness = report["seed_robustness"]
    expected_seeds = [int(seed) for seed in evaluation_config["robustness_seeds"]]

    assert robustness["seeds"] == expected_seeds
    assert set(robustness["per_seed"]) == {str(seed) for seed in expected_seeds}
    assert robustness["precision_mean"] >= evaluation_config["target_precision"] - 0.05
    assert robustness["precision_std"] <= 0.05
    assert robustness["recall_std"] <= 0.05
