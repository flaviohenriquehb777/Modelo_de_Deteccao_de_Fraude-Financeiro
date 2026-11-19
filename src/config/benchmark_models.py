"""Benchmark multi-modelo e busca restrita de hiperparametros (Fase 5).

Objetivo de negocio: precision ~0,96 mantendo recall > 0,80 (ver params.yaml).

Metodologia (HONESTA, identica a do modelo oficial):
- MESMO protocolo anti-leakage de `train_model.py`: StratifiedKFold
  (evaluation.cv_folds, seed fixa) -> probabilidades out-of-fold por familia ->
  calibracao global unica (monotonica, preserva a fronteira PR) -> selecao de
  threshold restrita (`maximize_precision_subject_to_recall`).
- Nenhuma estatistica de teste entra no treino: o Pipeline (feature engineering
  + preprocess + classificador) e ajustado apenas no treino de cada dobra.
- A busca de hiperparametros e RESTRITA e ANINHADA: RandomizedSearchCV com um
  scorer que maximiza precision sujeito a recall >= piso, rodando DENTRO de cada
  dobra externa. A metrica reportada e a do out-of-fold honesto (nunca
  `best_score_`), evitando o vies de selecao que reintroduziria o problema da
  Fase 1.

Este script e STANDALONE (nao faz parte do `dvc repro`): roda sob demanda para
gerar evidencia e nao penaliza o tempo da CI. A evidencia versionada e
`reports/model_benchmark.json` + `reports/model_benchmark.md`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import PROCESSED_DATA_PATH, PROJECT_ROOT
from src.config.train_model import (
    build_feature_engineer,
    build_preprocessor,
    build_threshold_trade_off,
    calibrate_out_of_fold_probabilities,
    compute_classification_metrics,
    compute_probability_metrics,
    confusion_matrix_counts,
    relative_posix_path,
    select_business_threshold,
    summarize_fold_metrics,
)

BENCHMARK_REPORT_PATH = PROJECT_ROOT / "reports" / "model_benchmark.json"
BENCHMARK_REPORT_MD_PATH = PROJECT_ROOT / "reports" / "model_benchmark.md"

FAMILIES = ("xgboost", "lightgbm", "random_forest", "logistic_regression")


def load_benchmark_context() -> tuple[dict, dict, dict, dict]:
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


def build_estimator(
    family: str, config: dict, class_ratio: float, random_state: int
) -> BaseEstimator:
    """Instancia o estimador de uma familia, resolvendo o peso de classe.

    `scale_pos_weight: auto` (boosting) e `class_weight: balanced` (RF/linear)
    tratam o desbalanceamento sem gerar amostras sinteticas.
    """
    params = {key: value for key, value in config.items() if value is not None}
    params.setdefault("random_state", random_state)

    if family == "xgboost":
        if params.get("scale_pos_weight") == "auto":
            params["scale_pos_weight"] = class_ratio
        return XGBClassifier(**params)
    if family == "lightgbm":
        if params.get("scale_pos_weight") == "auto":
            params["scale_pos_weight"] = class_ratio
        params.setdefault("verbose", -1)
        # `boost_from_average` inicializa o score pela media ponderada dos
        # rotulos. Com `scale_pos_weight` alto (desbalanceamento extremo) a media
        # ponderada tende a 0,5 e o LightGBM parte de um ponto degenerado que
        # satura as probabilidades e derruba a PR-AUC (~0,04 vs ~0,88 do
        # XGBoost). Desligar o boost restaura a fronteira precision-recall.
        if params.get("scale_pos_weight", 1.0) != 1.0:
            params.setdefault("boost_from_average", False)
        return LGBMClassifier(**params)
    if family == "random_forest":
        params.pop("scale_pos_weight", None)
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**params)
    if family == "logistic_regression":
        params.pop("scale_pos_weight", None)
        return LogisticRegression(**params)
    raise ValueError(f"Familia de modelo desconhecida: {family}")


def build_family_pipeline(
    family: str,
    config: dict,
    preprocess_config: dict,
    class_ratio: float,
    random_state: int,
) -> Pipeline:
    """Pipeline anti-leakage (features + preprocess + classificador) por familia.

    A calibracao NAO entra aqui: ela e ajustada uma unica vez sobre as
    probabilidades out-of-fold agregadas (ver `calibrate_out_of_fold_probabilities`).
    """
    return Pipeline(
        [
            ("features", build_feature_engineer(preprocess_config)),
            ("preprocess", build_preprocessor(preprocess_config)),
            ("classifier", build_estimator(family, config, class_ratio, random_state)),
        ]
    )


def build_search_space(family: str, class_ratio: float) -> dict:
    """Espaco de busca restrito por familia (chaves prefixadas por `classifier__`)."""
    spaces = {
        "xgboost": {
            "classifier__max_depth": [3, 5, 7, 9],
            "classifier__n_estimators": [200, 300, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__gamma": [0.0, 0.1, 0.5],
            "classifier__scale_pos_weight": [1.0, class_ratio],
        },
        "lightgbm": {
            "classifier__num_leaves": [31, 63, 127],
            "classifier__n_estimators": [200, 300, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__min_child_samples": [20, 50, 100],
            "classifier__scale_pos_weight": [1.0, class_ratio],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [8, 12, 16, None],
            "classifier__min_samples_leaf": [1, 5, 10],
            "classifier__max_features": ["sqrt", 0.5],
        },
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
    }
    return spaces[family]


def constrained_precision(
    y_true: np.ndarray, probabilities: np.ndarray, recall_floor: float
) -> float:
    """Maior precision entre os pontos da curva PR com recall >= piso.

    Retorna -1.0 quando nenhum ponto respeita o piso, sinalizando uma
    configuracao inviavel para a busca (que maximiza este valor).
    """
    precision_values, recall_values, _ = precision_recall_curve(y_true, probabilities)
    precision_values, recall_values = precision_values[:-1], recall_values[:-1]
    feasible = recall_values >= recall_floor
    if not np.any(feasible):
        return -1.0
    return float(precision_values[feasible].max())


def make_constrained_scorer(recall_floor: float):
    """Scorer customizado `scorer(estimator, X, y)` que respeita a restricao."""

    def scorer(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray) -> float:
        return constrained_precision(y, estimator.predict_proba(X)[:, 1], recall_floor)

    return scorer


def run_family_oof(
    X: pd.DataFrame,
    y: pd.Series,
    family: str,
    config: dict,
    splits: list,
    preprocess_config: dict,
    calibration_config: dict,
    class_ratio: float,
    random_state: int,
) -> np.ndarray:
    """Probabilidades out-of-fold calibradas de uma familia (uma passada honesta)."""
    oof_probabilities = np.zeros(len(y), dtype=float)
    for train_index, test_index in splits:
        pipeline = build_family_pipeline(
            family, config, preprocess_config, class_ratio, random_state
        )
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        oof_probabilities[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]
    return calibrate_out_of_fold_probabilities(y, oof_probabilities, calibration_config)


def summarize_oof(
    y: pd.Series,
    oof_probabilities: np.ndarray,
    splits: list,
    benchmark_evaluation: dict,
) -> dict:
    """Metricas honestas de um ponto operacional selecionado na fronteira OOF.

    O threshold e escolhido por `maximize_precision_subject_to_recall` sobre as
    probabilidades out-of-fold calibradas; as metricas por dobra sao avaliadas
    nesse ponto (e nao em 0.5), pois e a faixa em que o modelo opera.
    """
    threshold_calibration = select_business_threshold(
        y, oof_probabilities, benchmark_evaluation
    )
    operating_threshold = float(threshold_calibration["selected"]["threshold"])
    operating_predictions = (oof_probabilities >= operating_threshold).astype(int)

    fold_metrics = []
    for _, test_index in splits:
        fold_probabilities = oof_probabilities[test_index]
        fold_predictions = (fold_probabilities >= operating_threshold).astype(int)
        fold_metrics.append(
            {
                **compute_classification_metrics(y.iloc[test_index], fold_predictions),
                **compute_probability_metrics(y.iloc[test_index], fold_probabilities),
            }
        )

    fold_summary = summarize_fold_metrics(fold_metrics)
    return {
        "operating_threshold": operating_threshold,
        "cross_validation_mean_metrics": {
            name: values["mean"] for name, values in fold_summary.items()
        },
        "cross_validation_summary": fold_summary,
        "out_of_fold_operating_threshold_metrics": compute_classification_metrics(
            y, operating_predictions
        ),
        "out_of_fold_probability_metrics": compute_probability_metrics(
            y, oof_probabilities
        ),
        "out_of_fold_confusion_matrix": confusion_matrix_counts(y, operating_predictions),
        "threshold_calibration": threshold_calibration,
        "threshold_trade_off": build_threshold_trade_off(
            y, oof_probabilities, benchmark_evaluation
        ),
    }


def rank_families(families: list[dict], recall_floor: float) -> list[dict]:
    """Ordena as familias priorizando factibilidade da meta e precision honesta.

    Chave de ordenacao: (recall >= piso, precision out-of-fold, PR-AUC). Ou seja,
    primeiro as configuracoes que respeitam a restricao de negocio, depois a
    maior precision factivel e, por fim, a melhor fronteira livre de threshold.
    """

    def sort_key(entry: dict):
        operating = entry["out_of_fold_operating_threshold_metrics"]
        return (
            operating["recall"] >= recall_floor,
            operating["precision"],
            entry["out_of_fold_probability_metrics"]["pr_auc"],
        )

    return sorted(families, key=sort_key, reverse=True)


def run_restricted_search(
    X: pd.DataFrame,
    y: pd.Series,
    family: str,
    base_config: dict,
    preprocess_config: dict,
    calibration_config: dict,
    class_ratio: float,
    tuning_config: dict,
    recall_floor: float,
) -> dict:
    """Busca restrita ANINHADA: seleciona hiperparametros dentro de cada dobra.

    Cada dobra externa roda um RandomizedSearchCV (scorer restrito) apenas sobre
    o seu treino e refita o melhor modelo para prever a dobra externa. As
    probabilidades out-of-fold resultantes sao calibradas e avaliadas como
    qualquer outra familia -> metrica honesta, sem o vies de `best_score_`.
    """
    random_state = int(tuning_config.get("random_state", 42))
    outer_splitter = StratifiedKFold(
        n_splits=int(tuning_config["outer_cv_folds"]),
        shuffle=True,
        random_state=random_state,
    )
    search_space = build_search_space(family, class_ratio)

    oof_probabilities = np.zeros(len(y), dtype=float)
    selected_params = []
    for train_index, test_index in outer_splitter.split(X, y):
        pipeline = build_family_pipeline(
            family, base_config, preprocess_config, class_ratio, random_state
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=search_space,
            n_iter=int(tuning_config["n_iter"]),
            scoring=make_constrained_scorer(recall_floor),
            cv=int(tuning_config["inner_cv_folds"]),
            random_state=random_state,
            n_jobs=int(tuning_config.get("n_jobs", 4)),
            refit=True,
        )
        search.fit(X.iloc[train_index], y.iloc[train_index])
        selected_params.append(
            {key: _jsonify(value) for key, value in sorted(search.best_params_.items())}
        )
        oof_probabilities[test_index] = search.best_estimator_.predict_proba(
            X.iloc[test_index]
        )[:, 1]

    oof_probabilities = calibrate_out_of_fold_probabilities(
        y, oof_probabilities, calibration_config
    )
    return {
        "family": family,
        "search_space_size": int(np.prod([len(v) for v in search_space.values()])),
        "n_iter": int(tuning_config["n_iter"]),
        "outer_cv_folds": int(tuning_config["outer_cv_folds"]),
        "inner_cv_folds": int(tuning_config["inner_cv_folds"]),
        "selected_params_per_outer_fold": selected_params,
        "oof_probabilities": oof_probabilities,
    }


def _jsonify(value: object) -> object:
    """Converte valores de hiperparametros para tipos serializaveis em JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def evaluate_families(
    X: pd.DataFrame,
    y: pd.Series,
    benchmark_config: dict,
    evaluation_config: dict,
    preprocess_config: dict,
    calibration_config: dict,
) -> dict:
    """Executa o benchmark honesto de todas as familias candidatas."""
    random_state = int(benchmark_config["random_state"])
    recall_floor = float(benchmark_config["recall_floor"])
    class_ratio = float(y.value_counts()[0] / y.value_counts()[1])
    splits = list(
        StratifiedKFold(
            n_splits=int(evaluation_config["cv_folds"]),
            shuffle=bool(evaluation_config["shuffle"]),
            random_state=random_state,
        ).split(X, y)
    )
    benchmark_evaluation = {**evaluation_config, "target_recall": recall_floor}

    families = []
    timings = {}
    for family, config in benchmark_config["candidates"].items():
        started = time.perf_counter()
        oof_probabilities = run_family_oof(
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
        summary = summarize_oof(y, oof_probabilities, splits, benchmark_evaluation)
        timings[family] = round(time.perf_counter() - started, 2)
        families.append({"family": family, "config": config, **summary})
        operating = summary["out_of_fold_operating_threshold_metrics"]
        print(
            f"[benchmark] {family:<20} "
            f"PR-AUC {summary['out_of_fold_probability_metrics']['pr_auc']:.4f} | "
            f"P {operating['precision']:.4f} @ R {operating['recall']:.4f} | "
            f"{timings[family]:.1f}s",
            flush=True,
        )

    ranking = rank_families(families, recall_floor)
    return {
        "families": families,
        "ranking": [entry["family"] for entry in ranking],
        "winner": ranking[0]["family"],
        "timing_seconds": timings,
    }


def build_report(
    X: pd.DataFrame,
    y: pd.Series,
    benchmark_config: dict,
    evaluation_config: dict,
    preprocess_config: dict,
    calibration_config: dict,
    benchmark: dict,
    tuning: dict | None,
) -> dict:
    total_rows = int(len(y))
    positive_count = int(y.sum())
    return {
        "generated_by": "src/config/benchmark_models.py",
        "protocol": {
            "cv_strategy": "StratifiedKFold",
            "cv_folds": int(evaluation_config["cv_folds"]),
            "random_state": int(benchmark_config["random_state"]),
            "recall_floor": float(benchmark_config["recall_floor"]),
            "selection_rule": "maximize_precision_subject_to_recall",
            "calibration": calibration_config or {"enabled": False},
            "dataset": {
                "processed_data_path": relative_posix_path(PROCESSED_DATA_PATH),
                "row_count": total_rows,
                "feature_count": int(X.shape[1]),
                "positive_count": positive_count,
                "positive_rate": float(positive_count / total_rows),
            },
        },
        "benchmark": benchmark,
        "tuning": tuning,
    }


def render_markdown(report: dict) -> str:
    protocol = report["protocol"]
    dataset = protocol["dataset"]
    families = {entry["family"]: entry for entry in report["benchmark"]["families"]}

    lines = [
        "# Benchmark multi-modelo — Deteccao de Fraude (Credicard)",
        "",
        "> Gerado automaticamente por `src/config/benchmark_models.py`. Nao editar manualmente.",
        "",
        "## Protocolo honesto (anti-leakage)",
        (
            f"- {protocol['cv_strategy']} com {protocol['cv_folds']} dobras "
            f"(seed={protocol['random_state']})"
        ),
        f"- Regra de threshold: `{protocol['selection_rule']}` com piso de recall "
        f"{protocol['recall_floor']:.2f}",
        f"- Calibracao: {protocol['calibration'].get('method', 'sigmoid')} "
        f"(global, unica, sobre o out-of-fold)",
        f"- Dataset: {dataset['row_count']} linhas | {dataset['feature_count']} features | "
        f"{dataset['positive_count']} fraudes ({dataset['positive_rate']:.4%})",
        "",
        "## Ranking (maior precision factivel no piso de recall)",
        "| # | familia | PR-AUC | ROC-AUC | precision (OOF) | recall (OOF) | thr |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for position, family in enumerate(report["benchmark"]["ranking"], start=1):
        entry = families[family]
        operating = entry["out_of_fold_operating_threshold_metrics"]
        probability = entry["out_of_fold_probability_metrics"]
        lines.append(
            f"| {position} | {family} | {probability['pr_auc']:.4f} | "
            f"{probability['roc_auc']:.4f} | {operating['precision']:.4f} | "
            f"{operating['recall']:.4f} | {entry['operating_threshold']:.3f} |"
        )

    winner_trade_off = families[report["benchmark"]["ranking"][0]]["threshold_trade_off"]
    trade_off_header = " | ".join(f"piso {row['recall_floor']:.2f}" for row in winner_trade_off)
    lines += [
        "",
        f"Vencedor do benchmark: **{report['benchmark']['winner']}**.",
        "",
        "## Trade-off precision x recall por familia",
        f"| familia | {trade_off_header} |",
        "| --- | " + " | ".join("---" for _ in winner_trade_off) + " |",
    ]
    for family in report["benchmark"]["ranking"]:
        cells = []
        for row in families[family]["threshold_trade_off"]:
            cells.append("-" if row["precision"] is None else f"{row['precision']:.4f}")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")

    lines += ["", "## Tempo de execucao por familia (s)", ""]
    for family, seconds in report["benchmark"]["timing_seconds"].items():
        lines.append(f"- {family}: {seconds:.1f}s")

    tuning = report.get("tuning")
    lines += ["", "## Busca restrita (validacao aninhada)", ""]
    if tuning is None:
        lines.append("Nao executada nesta rodada.")
    else:
        operating = tuning["out_of_fold_operating_threshold_metrics"]
        probability = tuning["out_of_fold_probability_metrics"]
        lines += [
            f"- Familia tunada: **{tuning['family']}**",
            f"- Espaco de busca: {tuning['search_space_size']} combinacoes | "
            f"n_iter={tuning['n_iter']} | outer={tuning['outer_cv_folds']} | "
            f"inner={tuning['inner_cv_folds']}",
            f"- Metrica honesta (OOF aninhado): PR-AUC {probability['pr_auc']:.4f} | "
            f"ROC-AUC {probability['roc_auc']:.4f} | precision {operating['precision']:.4f} | "
            f"recall {operating['recall']:.4f} @ thr {tuning['operating_threshold']:.3f}",
            "",
            "### Hiperparametros selecionados por dobra externa",
            "",
            "| dobra | parametros |",
            "| --- | --- |",
        ]
        for index, params in enumerate(tuning["selected_params_per_outer_fold"], start=1):
            rendered = ", ".join(f"{key}={value}" for key, value in params.items())
            lines.append(f"| {index} | {rendered} |")

    lines.append("")
    return "\n".join(lines)


def save_report(report: dict) -> None:
    BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with BENCHMARK_REPORT_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(render_markdown(report))


def log_to_mlflow(report: dict) -> None:
    """Registra o benchmark no MLflow local (opt-in). Nunca derruba a execucao."""
    try:
        import mlflow
    except ImportError:
        print("[mlflow] mlflow indisponivel; registro ignorado.")
        return

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("benchmark_fase5")
    for entry in report["benchmark"]["families"]:
        with mlflow.start_run(run_name=f"benchmark_{entry['family']}"):
            mlflow.log_params({f"base_{k}": v for k, v in entry["config"].items()})
            mlflow.log_metric("pr_auc", entry["out_of_fold_probability_metrics"]["pr_auc"])
            mlflow.log_metric("roc_auc", entry["out_of_fold_probability_metrics"]["roc_auc"])
            operating = entry["out_of_fold_operating_threshold_metrics"]
            mlflow.log_metric("precision", operating["precision"])
            mlflow.log_metric("recall", operating["recall"])
            mlflow.log_metric("operating_threshold", entry["operating_threshold"])
    mlflow.log_artifact(str(BENCHMARK_REPORT_PATH))
    mlflow.log_artifact(str(BENCHMARK_REPORT_MD_PATH))
    print("[mlflow] benchmark registrado no experimento 'benchmark_fase5'.")


def _strip_runtime_fields(tuning: dict) -> dict:
    return {key: value for key, value in tuning.items() if key != "oof_probabilities"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-modelo (Fase 5).")
    parser.add_argument(
        "--family",
        choices=[*FAMILIES, "auto"],
        default=None,
        help="Familia a tunar (default: benchmark.tuning.family).",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Roda apenas o benchmark das configuracoes-base (sem busca restrita).",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Registra o benchmark no MLflow local (mlruns/, git-ignored).",
    )
    args = parser.parse_args()

    benchmark_config, evaluation_config, preprocess_config, calibration_config = (
        load_benchmark_context()
    )
    tuning_config = dict(benchmark_config.get("tuning", {}))

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    class_ratio = float(y.value_counts()[0] / y.value_counts()[1])
    recall_floor = float(benchmark_config["recall_floor"])
    random_state = int(benchmark_config["random_state"])

    benchmark = evaluate_families(
        X, y, benchmark_config, evaluation_config, preprocess_config, calibration_config
    )

    tuning = None
    if not args.no_search:
        requested_family = args.family or tuning_config.get("family", "auto")
        search_family = (
            benchmark["winner"] if requested_family in (None, "auto") else requested_family
        )
        print(f"[search] busca restrita restrita na familia: {search_family}", flush=True)
        started = time.perf_counter()
        raw_tuning = run_restricted_search(
            X,
            y,
            search_family,
            benchmark_config["candidates"][search_family],
            preprocess_config,
            calibration_config,
            class_ratio,
            {**tuning_config, "random_state": random_state},
            recall_floor,
        )
        oof_probabilities = raw_tuning.pop("oof_probabilities")
        splits = list(
            StratifiedKFold(
                n_splits=int(evaluation_config["cv_folds"]),
                shuffle=bool(evaluation_config["shuffle"]),
                random_state=random_state,
            ).split(X, y)
        )
        tuning = {
            **raw_tuning,
            **summarize_oof(
                y,
                oof_probabilities,
                splits,
                {**evaluation_config, "target_recall": recall_floor},
            ),
            "timing_seconds": round(time.perf_counter() - started, 2),
        }
        operating = tuning["out_of_fold_operating_threshold_metrics"]
        print(
            f"[search] {search_family} | PR-AUC "
            f"{tuning['out_of_fold_probability_metrics']['pr_auc']:.4f} | "
            f"P {operating['precision']:.4f} @ R {operating['recall']:.4f} | "
            f"{tuning['timing_seconds']:.1f}s",
            flush=True,
        )

    report = build_report(
        X,
        y,
        benchmark_config,
        evaluation_config,
        preprocess_config,
        calibration_config,
        benchmark,
        None if tuning is None else _strip_runtime_fields(tuning),
    )
    save_report(report)
    print(f"[report] {BENCHMARK_REPORT_PATH} e {BENCHMARK_REPORT_MD_PATH}")

    if args.mlflow:
        log_to_mlflow(report)


if __name__ == "__main__":
    main()
