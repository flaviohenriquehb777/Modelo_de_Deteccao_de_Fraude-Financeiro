"""Testes leves da logica pura do benchmark multi-modelo (Fase 5).

Nao treinam modelos: validam as funcoes de decisao (scorer restrito, ranking,
espaco de busca e renderizacao do relatorio) que sustentam a evidencia honesta.
"""

import numpy as np

from src.config.benchmark_models import (
    build_estimator,
    build_search_space,
    constrained_precision,
    rank_families,
    render_markdown,
)


def test_constrained_precision_returns_minus_one_when_infeasible():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    assert constrained_precision(y, probabilities, 1.5) == -1.0


def test_constrained_precision_attains_one_for_separable_scores():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    assert constrained_precision(y, probabilities, 0.5) == 1.0


def _family(family: str, recall: float, precision: float, pr_auc: float) -> dict:
    return {
        "family": family,
        "config": {},
        "operating_threshold": 0.5,
        "out_of_fold_operating_threshold_metrics": {"precision": precision, "recall": recall},
        "out_of_fold_probability_metrics": {"pr_auc": pr_auc, "roc_auc": 0.98},
        "threshold_trade_off": [],
    }


def test_rank_families_prefers_feasible_then_precision_then_pr_auc():
    families = [
        _family("a", recall=0.70, precision=0.99, pr_auc=0.90),  # inviavel na restricao
        _family("b", recall=0.85, precision=0.95, pr_auc=0.80),  # factivel
        _family("c", recall=0.90, precision=0.95, pr_auc=0.70),  # factivel, empate -> PR-AUC
    ]

    ranking = rank_families(families, recall_floor=0.80)

    assert [entry["family"] for entry in ranking] == ["b", "c", "a"]


def test_lightgbm_disables_boost_from_average_under_class_weighting():
    """Regressao: boost_from_average + scale_pos_weight alto degenera a PR-AUC."""
    weighted = build_estimator("lightgbm", {"scale_pos_weight": "auto"}, 577.0, 42)
    assert weighted.get_params()["boost_from_average"] is False

    unweighted = build_estimator("lightgbm", {"scale_pos_weight": 1.0}, 577.0, 42)
    # Ausencia da chave significa que o default do LightGBM (True) foi mantido.
    assert unweighted.get_params().get("boost_from_average", True) is True


def test_build_search_space_is_prefixed_for_pipeline_step():
    space = build_search_space("xgboost", class_ratio=577.0)

    assert all(key.startswith("classifier__") for key in space)
    assert 577.0 in space["classifier__scale_pos_weight"]


def test_render_markdown_surfaces_ranking_and_winner():
    families = [
        _family("xgboost", recall=0.8028, precision=0.9611, pr_auc=0.8573),
        _family("logistic_regression", recall=0.80, precision=0.60, pr_auc=0.55),
    ]
    report = {
        "protocol": {
            "cv_strategy": "StratifiedKFold",
            "cv_folds": 10,
            "random_state": 42,
            "selection_rule": "maximize_precision_subject_to_recall",
            "recall_floor": 0.80,
            "calibration": {"method": "sigmoid"},
            "dataset": {
                "row_count": 284807,
                "feature_count": 33,
                "positive_count": 492,
                "positive_rate": 0.0017,
            },
        },
        "benchmark": {
            "families": families,
            "ranking": ["xgboost", "logistic_regression"],
            "winner": "xgboost",
            "timing_seconds": {"xgboost": 1.0, "logistic_regression": 0.5},
        },
        "tuning": None,
    }

    markdown = render_markdown(report)

    assert "## Ranking" in markdown
    assert "Vencedor do benchmark: **xgboost**" in markdown
    assert "logistic_regression" in markdown
