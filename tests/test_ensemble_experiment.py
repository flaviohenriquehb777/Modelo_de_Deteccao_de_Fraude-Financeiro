import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config.ensemble_experiment import (
    build_verdict,
    logit_transform,
    render_markdown,
    simple_blend,
    stack_out_of_fold,
)


def _splits(y: pd.Series, folds: int = 5, seed: int = 42) -> list:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((len(y), 1)), y))


def _strategy(
    name: str,
    strategy_type: str,
    pr_auc: float,
    precision_at_floor: float,
    simultaneously_achievable: bool = False,
) -> dict:
    return {
        "name": name,
        "type": strategy_type,
        "operating_threshold": 0.566,
        "out_of_fold_probability_metrics": {"pr_auc": pr_auc, "roc_auc": 0.9804},
        "out_of_fold_operating_threshold_metrics": {"precision": 0.9611, "recall": 0.8028},
        "threshold_trade_off": [],
        "aspirational_target": {
            "target_precision": 0.96,
            "target_recall": 0.82,
            "blocking": False,
            "best_feasible": {
                "threshold": 0.1859,
                "precision": precision_at_floor,
                "recall": 0.8211,
            },
            "precision_gap": 0.96 - precision_at_floor,
            "simultaneously_achievable": simultaneously_achievable,
        },
    }


def test_simple_blend_is_row_mean():
    matrix = np.array([[0.2, 0.4], [0.8, 0.6], [0.0, 1.0]])

    blended = simple_blend(matrix)

    np.testing.assert_allclose(blended, [0.3, 0.7, 0.5])


def test_stack_out_of_fold_learns_to_combine_complementary_signals():
    """O meta-modelo deve superar o blend quando os bases sao complementares."""
    rng = np.random.default_rng(42)
    n_samples = 4000
    y = pd.Series(rng.integers(0, 2, size=n_samples))
    signal = y.to_numpy() * 0.5 + rng.normal(0, 0.25, size=n_samples)
    p_a = (signal - signal.min()) / (signal.max() - signal.min())
    p_b = 1.0 - p_a
    meta = np.column_stack([p_a, p_b])

    blend_auc = roc_auc_score(y, simple_blend(meta))
    stacked = stack_out_of_fold(meta, y, _splits(y), random_state=42)
    stacked_auc = roc_auc_score(y, stacked)

    assert stacked.shape[0] == n_samples
    assert np.all((stacked >= 0.0) & (stacked <= 1.0))
    # O blend de sinais complementares se cancela (ROC-AUC ~ 0.5); o stacking
    # aprende os pesos (sinal x anti-sinal) e recupera o poder discriminativo.
    assert blend_auc < 0.55
    assert stacked_auc > 0.8


def test_stack_out_of_fold_is_deterministic():
    rng = np.random.default_rng(7)
    n_samples = 1000
    y = pd.Series(rng.integers(0, 2, size=n_samples))
    meta = rng.random((n_samples, 3))
    splits = _splits(y)

    first = stack_out_of_fold(meta, y, splits, random_state=42)
    second = stack_out_of_fold(meta, y, splits, random_state=42)

    np.testing.assert_allclose(first, second)


def test_logit_transform_is_finite_and_monotonic():
    probabilities = np.array([[0.0, 0.5], [1.0, 0.9]])

    logits = logit_transform(probabilities, epsilon=1e-6)

    assert np.all(np.isfinite(logits))
    assert logits[1, 0] > logits[0, 0]
    assert logits[1, 1] > logits[0, 1]


def test_build_verdict_flags_ensemble_frontier_move_only_when_both_gain():
    singles = [_strategy("lightgbm", "single", pr_auc=0.8596, precision_at_floor=0.9140)]
    # Ensemble melhora PR-AUC mas PIORA a precision no piso -> nao move a fronteira.
    worsens = [
        _strategy("stacking_logreg", "stacking", pr_auc=0.8700, precision_at_floor=0.9000)
    ]
    verdict_worse = build_verdict([*singles, *worsens], aspirational_precision=0.96)
    assert verdict_worse["ensemble_moves_frontier"] is False

    # Ensemble melhora AMBOS -> move a fronteira.
    improves = [
        _strategy("stacking_logreg", "stacking", pr_auc=0.8700, precision_at_floor=0.9300)
    ]
    verdict_better = build_verdict([*singles, *improves], aspirational_precision=0.96)
    assert verdict_better["ensemble_moves_frontier"] is True
    assert verdict_better["ensemble_delta_pr_auc_vs_best_single"] > 0


def test_build_verdict_reports_gap_and_non_achievability():
    strategies = [
        _strategy("lightgbm", "single", pr_auc=0.8596, precision_at_floor=0.9140),
        _strategy("blend_mean", "ensemble", pr_auc=0.8600, precision_at_floor=0.9100),
    ]

    verdict = build_verdict(strategies, aspirational_precision=0.96)

    assert verdict["simultaneously_achievable"] is False
    assert verdict["best_feasible_precision_at_aspirational_recall"] == 0.9140
    assert round(verdict["gap_to_aspirational_precision"], 4) == round(0.96 - 0.9140, 4)
    assert verdict["best_strategy_overall"] == "blend_mean"


def test_render_markdown_surfaces_verdict():
    report = {
        "protocol": {
            "cv_strategy": "StratifiedKFold",
            "cv_folds": 10,
            "random_state": 42,
            "selection_rule": "maximize_precision_subject_to_recall",
            "calibration": {"method": "sigmoid"},
            "aspirational": {"precision": 0.96, "recall": 0.82, "blocking": False},
            "dataset": {
                "row_count": 284807,
                "feature_count": 33,
                "positive_count": 492,
                "positive_rate": 0.0017,
            },
        },
        "strategies": [
            _strategy("lightgbm", "single", pr_auc=0.8596, precision_at_floor=0.9140),
            _strategy("stacking_logreg", "stacking", pr_auc=0.8700, precision_at_floor=0.9300),
        ],
        "verdict": build_verdict(
            [
                _strategy("lightgbm", "single", pr_auc=0.8596, precision_at_floor=0.9140),
                _strategy(
                    "stacking_logreg", "stacking", pr_auc=0.8700, precision_at_floor=0.9300
                ),
            ],
            aspirational_precision=0.96,
        ),
        "notes": ["nota de teste"],
    }

    markdown = render_markdown(report)

    assert "Experimento de ensemble/stacking" in markdown
    assert "stacking_logreg" in markdown
    assert "Ensemble move a fronteira?" in markdown
    assert "nota de teste" in markdown
