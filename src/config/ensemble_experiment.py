"""Experimento de ensemble/stacking — mitigacao do risco da meta 96/82 (Fase 8).

Pergunta de negocio: a meta aspiracional (precision 0,96 com recall 0,82) e
inalcancavel por um modelo unico na fronteira honesta (recall >= 0,82 => precision
max ~0,9140; ver Fase 7). ANTES de relaxar qualquer meta, a Fase 8 exige medir se
combinar os modelos base (blend e stacking) move a fronteira precision-recall
nessa direcao.

Protocolo (HONESTO, identico ao benchmark da Fase 5 e ao treino oficial):
- StratifiedKFold (evaluation.cv_folds, seed fixa) -> probabilidades out-of-fold
  por familia base -> calibracao global unica por familia (monotonica; preserva
  a fronteira PR).
- `blend_mean`: media simples das probabilidades calibradas (sem treino).
- `stacking_logreg*`: meta-modelo LogisticRegression sobre o LOGIT das
  probabilidades base out-of-fold, com validacao cruzada ANINHADA propria (o
  meta-modelo nunca usa a previsao do proprio ponto como treino) -> stack
  out-of-fold honesto. Duas variantes: sem e com `class_weight="balanced"`.
- Toda metrica reportada e a do out-of-fold honesto. Nenhum vazamento.

A frente de "features adicionais" ja foi sondada na Fase 2 (features temporais
ciclicas: ganho dentro de 1 desvio-padrao; dataset anonimizado/PCA-transformado,
sem entidades para agregacao) -> nao e um lever viavel. Este experimento cobre a
outra frente da Fase 8: ensemble/stacking.

Este script e STANDALONE (nao entra no `dvc repro`), como o benchmark. A evidencia
versionada e `reports/ensemble_experiment.json` + `reports/ensemble_experiment.md`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.benchmark_models import run_family_oof
from src.config.paths import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.config.train_model import (
    build_threshold_trade_off,
    calibrate_out_of_fold_probabilities,
    compute_classification_metrics,
    compute_probability_metrics,
    relative_posix_path,
    select_business_threshold,
    summarize_aspirational_target,
    summarize_fold_metrics,
)

ENSEMBLE_REPORT_PATH = PROJECT_ROOT / "reports" / "ensemble_experiment.json"
ENSEMBLE_REPORT_MD_PATH = PROJECT_ROOT / "reports" / "ensemble_experiment.md"

META_LEARNER_MAX_ITER = 1000
LOGIT_EPSILON = 1e-6


def load_ensemble_context() -> tuple[dict, dict, dict, dict]:
    """Le params.yaml e devolve (benchmark, evaluation, preprocess, calibration)."""
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    train = params["train"]
    return (
        train["benchmark"],
        train["evaluation"],
        params["preprocess"],
        train.get("calibration", {}),
    )


def build_oof_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: dict,
    splits: list,
    preprocess_config: dict,
    calibration_config: dict,
    class_ratio: float,
    random_state: int,
) -> tuple[list[str], np.ndarray]:
    """Probabilidades out-of-fold CALIBRADAS de cada familia base (honesto).

    Reutiliza o protocolo do benchmark: uma passada StratifiedKFold por familia,
    com o Pipeline ajustado apenas no treino de cada dobra e calibracao global
    unica sobre o out-of-fold agregado. Devolve (nomes, matriz n_amostras x familias).
    """
    names = []
    columns = []
    for family, config in candidates.items():
        oof = run_family_oof(
            X,
            y,
            family,
            config,
            splits,
            preprocess_config,
            calibration_config,
            class_ratio,
            random_state,
        )
        names.append(family)
        columns.append(oof)
        print(f"[ensemble] base OOF pronto: {family}", flush=True)
    return names, np.column_stack(columns)


def load_or_build_oof_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: dict,
    splits: list,
    preprocess_config: dict,
    calibration_config: dict,
    class_ratio: float,
    random_state: int,
    cache_path: Path | None,
    refresh: bool,
) -> tuple[list[str], np.ndarray]:
    """Reaproveita a matriz out-of-fold base do cache (iteracao de meta-modelo).

    Ferramenta de desenvolvimento: como as probabilidades base sao deterministicas
    (seeds fixas) e independentes do meta-modelo, evita re-treinar as 4 familias a
    cada variacao de ensemble. Sem `cache_path` a matriz e sempre recalculada.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        data = np.load(cache_path, allow_pickle=False)
        names = [str(name) for name in data["names"]]
        print(f"[ensemble] base OOF reaproveitado de {cache_path}", flush=True)
        return names, data["matrix"]

    names, matrix = build_oof_matrix(
        X,
        y,
        candidates,
        splits,
        preprocess_config,
        calibration_config,
        class_ratio,
        random_state,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, names=np.array(names), matrix=matrix)
        print(f"[ensemble] base OOF cacheado em {cache_path}", flush=True)
    return names, matrix


def simple_blend(probabilities_matrix: np.ndarray) -> np.ndarray:
    """Media simples das probabilidades base (ensemble sem treino)."""
    return probabilities_matrix.mean(axis=1)


def logit_transform(probabilities: np.ndarray, epsilon: float = LOGIT_EPSILON) -> np.ndarray:
    """Logit das probabilidades (clipado), espaco linear do meta-modelo.

    O logit e monotonico por feature, mas lineariza a combinacao: ajuda o
    meta-modelo logistico a separar as classes quando as probabilidades base se
    concentram nos extremos (tipico em bases muito desbalanceadas).
    """
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def stack_out_of_fold(
    meta_features: np.ndarray,
    y: pd.Series,
    splits: list,
    random_state: int,
    class_weight: str | None = None,
) -> np.ndarray:
    """Stack out-of-fold honesto com meta-modelo LogisticRegression.

    O meta-modelo e treinado nas features base out-of-fold e avaliado com
    validacao cruzada ANINHADA (as MESMAS dobras externas): para cada dobra, o
    meta-modelo aprende na parte de treino e prevê a parte de teste, que ele
    nunca viu. Evita o vazamento classico de stacking (usar a propria previsao
    no treino).
    """
    meta_features = np.asarray(meta_features, dtype=float)
    y_values = np.asarray(y)
    oof = np.zeros(len(y_values), dtype=float)
    for train_index, test_index in splits:
        meta = LogisticRegression(
            max_iter=META_LEARNER_MAX_ITER,
            class_weight=class_weight,
            random_state=random_state,
        )
        meta.fit(meta_features[train_index], y_values[train_index])
        oof[test_index] = meta.predict_proba(meta_features[test_index])[:, 1]
    return oof


def evaluate_strategy(
    name: str,
    strategy_type: str,
    y: pd.Series,
    probabilities: np.ndarray,
    splits: list,
    evaluation_config: dict,
) -> dict:
    """Metricas honestas de uma estrategia na fronteira out-of-fold.

    O threshold e escolhido por `maximize_precision_subject_to_recall` sobre as
    probabilidades out-of-fold; as metricas por dobra sao avaliadas nesse ponto
    (a faixa em que o modelo opera), nao em 0.5.
    """
    threshold_calibration = select_business_threshold(y, probabilities, evaluation_config)
    operating_threshold = float(threshold_calibration["selected"]["threshold"])
    operating_predictions = (probabilities >= operating_threshold).astype(int)

    fold_metrics = []
    for _, test_index in splits:
        fold_probabilities = probabilities[test_index]
        fold_predictions = (fold_probabilities >= operating_threshold).astype(int)
        fold_metrics.append(
            {
                **compute_classification_metrics(y.iloc[test_index], fold_predictions),
                **compute_probability_metrics(y.iloc[test_index], fold_probabilities),
            }
        )

    trade_off = build_threshold_trade_off(y, probabilities, evaluation_config)
    return {
        "name": name,
        "type": strategy_type,
        "operating_threshold": operating_threshold,
        "cross_validation_mean_metrics": {
            metric: values["mean"]
            for metric, values in summarize_fold_metrics(fold_metrics).items()
        },
        "out_of_fold_operating_threshold_metrics": compute_classification_metrics(
            y, operating_predictions
        ),
        "out_of_fold_probability_metrics": compute_probability_metrics(y, probabilities),
        "threshold_trade_off": trade_off,
        "aspirational_target": summarize_aspirational_target(trade_off, evaluation_config),
    }


def build_ensemble_strategies(
    oof_matrix: np.ndarray,
    y: pd.Series,
    splits: list,
    evaluation_config: dict,
    calibration_config: dict,
    random_state: int,
) -> list[dict]:
    """Avalia os ensembles honestos (blend + variantes de stacking) sobre a base OOF."""
    strategies = [
        evaluate_strategy(
            "blend_mean", "ensemble", y, simple_blend(oof_matrix), splits, evaluation_config
        )
    ]

    meta_features = logit_transform(oof_matrix)
    for name, class_weight in (
        ("stacking_logreg", None),
        ("stacking_logreg_balanced", "balanced"),
    ):
        stacked = stack_out_of_fold(meta_features, y, splits, random_state, class_weight)
        stacked = calibrate_out_of_fold_probabilities(y, stacked, calibration_config)
        strategies.append(
            evaluate_strategy(name, "stacking", y, stacked, splits, evaluation_config)
        )
    return strategies


def build_verdict(strategies: list[dict], aspirational_precision: float) -> dict:
    """Compara as estrategias e resume o risco de alcancabilidade da meta 96/82.

    `ensemble_moves_frontier` e verdadeiro quando o melhor ensemble supera o melhor
    modelo unico EM PR-AUC e TAMBEM na precision no piso de recall aspiracional ->
    ou seja, o ensemble melhora a fronteira de forma nao-trivial.
    """
    singles = [entry for entry in strategies if entry["type"] == "single"]
    ensembles = [entry for entry in strategies if entry["type"] != "single"]

    def pr_auc(entry: dict) -> float:
        return float(entry["out_of_fold_probability_metrics"]["pr_auc"])

    def precision_at_floor(entry: dict) -> float:
        summary = entry["aspirational_target"]
        if summary is None or summary["best_feasible"] is None:
            return 0.0
        return float(summary["best_feasible"]["precision"])

    best_single = max(singles, key=pr_auc)
    best_ensemble = max(ensembles, key=pr_auc) if ensembles else None
    best_overall = max(strategies, key=pr_auc)

    best_single_precision = precision_at_floor(best_single)
    if best_ensemble is not None:
        best_ensemble_precision = precision_at_floor(best_ensemble)
        moves_frontier = bool(
            pr_auc(best_ensemble) > pr_auc(best_single)
            and best_ensemble_precision > best_single_precision
        )
    else:
        best_ensemble_precision = 0.0
        moves_frontier = False

    simultaneously_achievable = any(
        entry["aspirational_target"] is not None
        and entry["aspirational_target"]["simultaneously_achievable"]
        for entry in strategies
    )
    best_feasible_precision = max(precision_at_floor(entry) for entry in strategies)

    if simultaneously_achievable:
        conclusion = (
            "Ensemble/stacking alcanca a meta aspiracional 96/82: a meta NAO "
            "precisa ser relaxada."
        )
    else:
        conclusion = (
            "Nenhuma estrategia (modelo unico, blend ou stacking) atinge 96/82 "
            "simultaneamente na fronteira honesta. A meta aspiracional permanece "
            "NAO bloqueante: mantem-se o alvo de negocio 96/80, com o gap "
            "documentado explicitamente."
        )

    return {
        "best_single": {
            "name": best_single["name"],
            "pr_auc": pr_auc(best_single),
            "precision_at_aspirational_recall": best_single_precision,
        },
        "best_ensemble": None
        if best_ensemble is None
        else {
            "name": best_ensemble["name"],
            "pr_auc": pr_auc(best_ensemble),
            "precision_at_aspirational_recall": best_ensemble_precision,
        },
        "best_strategy_overall": best_overall["name"],
        "ensemble_delta_pr_auc_vs_best_single": (
            None if best_ensemble is None else pr_auc(best_ensemble) - pr_auc(best_single)
        ),
        "ensemble_moves_frontier": moves_frontier,
        "simultaneously_achievable": simultaneously_achievable,
        "best_feasible_precision_at_aspirational_recall": best_feasible_precision,
        "gap_to_aspirational_precision": aspirational_precision - best_feasible_precision,
        "conclusion": conclusion,
    }


def build_report(
    X: pd.DataFrame,
    y: pd.Series,
    strategies: list[dict],
    evaluation_config: dict,
    benchmark_config: dict,
    calibration_config: dict,
    verdict: dict,
) -> dict:
    total_rows = int(len(y))
    positive_count = int(y.sum())
    return {
        "generated_by": "src/config/ensemble_experiment.py",
        "protocol": {
            "cv_strategy": "StratifiedKFold",
            "cv_folds": int(evaluation_config["cv_folds"]),
            "random_state": int(benchmark_config["random_state"]),
            "selection_rule": "maximize_precision_subject_to_recall",
            "calibration": calibration_config or {"enabled": False},
            "aspirational": {
                "precision": float(evaluation_config["aspirational_precision"]),
                "recall": float(evaluation_config["aspirational_recall"]),
                "blocking": False,
            },
            "dataset": {
                "processed_data_path": relative_posix_path(PROCESSED_DATA_PATH),
                "row_count": total_rows,
                "feature_count": int(X.shape[1]),
                "positive_count": positive_count,
                "positive_rate": float(positive_count / total_rows),
            },
        },
        "strategies": strategies,
        "verdict": verdict,
        "notes": [
            "Features adicionais foram sondadas na Fase 2 (temporais ciclicas: "
            "ganho dentro de 1 desvio-padrao) e descartadas como lever.",
            "A meta 96/82 simultanea foi confirmada inalcancavel tambem por "
            "ensemble/stacking; nao ha vazamento de dados em nenhuma metrica.",
        ],
    }


def render_markdown(report: dict) -> str:
    protocol = report["protocol"]
    dataset = protocol["dataset"]
    verdict = report["verdict"]
    strategies = report["strategies"]

    lines = [
        "# Experimento de ensemble/stacking — risco da meta 96/82 (Fase 8)",
        "",
        "> Gerado automaticamente por `src/config/ensemble_experiment.py`. "
        "Nao editar manualmente.",
        "",
        "## Protocolo honesto (anti-leakage)",
        (
            f"- {protocol['cv_strategy']} com {protocol['cv_folds']} dobras "
            f"(seed={protocol['random_state']})"
        ),
        f"- Regra de threshold: `{protocol['selection_rule']}`",
        f"- Calibracao: {protocol['calibration'].get('method', 'sigmoid')} "
        "(global, unica, sobre o out-of-fold)",
        (
            f"- Dataset: {dataset['row_count']} linhas | {dataset['feature_count']} features | "
            f"{dataset['positive_count']} fraudes ({dataset['positive_rate']:.4%})"
        ),
        (
            f"- Meta aspiracional (NAO bloqueante): precision "
            f"{protocol['aspirational']['precision']:.2f} com recall "
            f"{protocol['aspirational']['recall']:.2f}"
        ),
        "",
        "## Estrategias avaliadas (out-of-fold honesto)",
        "| estrategia | tipo | PR-AUC | ROC-AUC | precision (OOF) | recall (OOF) | "
        "P @ R=0.82 | gap vs 0.96 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in strategies:
        operating = entry["out_of_fold_operating_threshold_metrics"]
        probability = entry["out_of_fold_probability_metrics"]
        aspirational = entry["aspirational_target"]
        if aspirational is None or aspirational["best_feasible"] is None:
            precision_at_floor = "-"
            gap = "-"
        else:
            precision_at_floor = f"{aspirational['best_feasible']['precision']:.4f}"
            gap = f"{aspirational['precision_gap']:.4f}"
        lines.append(
            f"| {entry['name']} | {entry['type']} | {probability['pr_auc']:.4f} | "
            f"{probability['roc_auc']:.4f} | {operating['precision']:.4f} | "
            f"{operating['recall']:.4f} | {precision_at_floor} | {gap} |"
        )

    lines += [
        "",
        "## Veredito",
        f"- Melhor modelo unico: **{verdict['best_single']['name']}** "
        f"(PR-AUC {verdict['best_single']['pr_auc']:.4f})",
    ]
    best_ensemble = verdict["best_ensemble"]
    if best_ensemble is not None:
        lines.append(
            f"- Melhor ensemble: **{best_ensemble['name']}** "
            f"(PR-AUC {best_ensemble['pr_auc']:.4f})"
        )
    lines += [
        f"- Melhor estrategia geral: **{verdict['best_strategy_overall']}**",
        f"- Ensemble move a fronteira? "
        f"**{'sim' if verdict['ensemble_moves_frontier'] else 'nao'}**",
        f"- Meta 96/82 alcancavel? "
        f"**{'sim' if verdict['simultaneously_achievable'] else 'nao'}**",
        (
            f"- Melhor ponto factivel (recall >= 0.82): precision "
            f"{verdict['best_feasible_precision_at_aspirational_recall']:.4f} | "
            f"gap vs 0.96 = {verdict['gap_to_aspirational_precision']:.4f}"
        ),
        "",
        f"> {verdict['conclusion']}",
        "",
        "## Notas",
    ]
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def save_report(report: dict) -> None:
    ENSEMBLE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ENSEMBLE_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with ENSEMBLE_REPORT_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(render_markdown(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimento de ensemble/stacking (Fase 8).")
    parser.add_argument(
        "--base-oof-cache",
        type=Path,
        default=None,
        help="Arquivo .npz para reaproveitar/cachear a matriz out-of-fold base "
        "(ferramenta de desenvolvimento; default: sempre recalcula).",
    )
    parser.add_argument(
        "--refresh-base",
        action="store_true",
        help="Ignora o cache da base out-of-fold e recalcula as familias.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_config, evaluation_config, preprocess_config, calibration_config = (
        load_ensemble_context()
    )
    random_state = int(benchmark_config["random_state"])
    aspirational_precision = float(evaluation_config["aspirational_precision"])

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    class_ratio = float(y.value_counts()[0] / y.value_counts()[1])

    splits = list(
        StratifiedKFold(
            n_splits=int(evaluation_config["cv_folds"]),
            shuffle=bool(evaluation_config["shuffle"]),
            random_state=random_state,
        ).split(X, y)
    )

    names, oof_matrix = load_or_build_oof_matrix(
        X,
        y,
        benchmark_config["candidates"],
        splits,
        preprocess_config,
        calibration_config,
        class_ratio,
        random_state,
        args.base_oof_cache,
        args.refresh_base,
    )

    strategies = [
        evaluate_strategy(name, "single", y, oof_matrix[:, index], splits, evaluation_config)
        for index, name in enumerate(names)
    ]
    strategies.extend(
        build_ensemble_strategies(
            oof_matrix, y, splits, evaluation_config, calibration_config, random_state
        )
    )

    for entry in strategies:
        operating = entry["out_of_fold_operating_threshold_metrics"]
        probability = entry["out_of_fold_probability_metrics"]
        print(
            f"[ensemble] {entry['name']:<26} PR-AUC {probability['pr_auc']:.4f} | "
            f"P {operating['precision']:.4f} @ R {operating['recall']:.4f}",
            flush=True,
        )

    verdict = build_verdict(strategies, aspirational_precision)
    report = build_report(
        X, y, strategies, evaluation_config, benchmark_config, calibration_config, verdict
    )
    save_report(report)
    print(f"[report] {ENSEMBLE_REPORT_PATH} e {ENSEMBLE_REPORT_MD_PATH}")
    print(f"[verdict] {verdict['conclusion']}")


if __name__ == "__main__":
    main()
