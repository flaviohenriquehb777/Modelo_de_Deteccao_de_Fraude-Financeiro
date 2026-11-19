import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from xgboost import XGBClassifier

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.feature_engineering import TemporalFeatureEngineer
from src.config.paths import (
    CONFUSION_MATRIX_PATH,
    EVALUATION_PLOTS_DIR,
    EVALUATION_REPORT_MD_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_METADATA_PATH,
    PIPELINE_MODEL_PATH,
    PR_CURVE_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
)

DEFAULT_THRESHOLD = 0.5
FOLD_METRIC_ORDER = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def relative_posix_path(path: Path) -> str:
    """Caminho relativo ao projeto, com separador '/' (portavel em relatorios)."""
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def load_training_config() -> tuple[dict, dict, dict, dict]:
    params_path = PROJECT_ROOT / "params.yaml"
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return (
        params["train"]["evaluation"],
        params["train"]["acceptance"],
        params["train"]["xgboost"],
        params["preprocess"],
    )


def build_model_config(base_config: dict, y: pd.Series) -> dict:
    model_config = dict(base_config)
    if model_config.get("scale_pos_weight") == "auto":
        class_ratio = y.value_counts()[0] / y.value_counts()[1]
        model_config["scale_pos_weight"] = float(class_ratio)
    return model_config


def build_preprocessor(preprocess_config: dict) -> ColumnTransformer:
    """Cria o pre-processador ajustavel por fold (anti-leakage).

    Os transformadores deixam de ser ajustados sobre o dataset inteiro e passam
    a ser ajustados apenas nos dados de treino de cada fold, dentro do Pipeline.
    """
    transformers = []
    if preprocess_config.get("amount_transform") == "yeo-johnson":
        transformers.append(("amount", PowerTransformer(method="yeo-johnson"), ["Amount"]))
    if preprocess_config.get("time_scaler") == "minmax":
        transformers.append(("time", MinMaxScaler(), ["Time"]))
    return ColumnTransformer(transformers, remainder="passthrough")


def build_feature_engineer(preprocess_config: dict) -> object:
    """Cria a etapa de engenharia de features (fold-safe) do Pipeline.

    As features derivadas sao calculadas de forma deterministica por linha, sem
    ajuste a partir dos dados, portanto nao vazam informacao entre os folds.
    """
    feature_config = preprocess_config.get("feature_engineering", {}) or {}
    if feature_config.get("temporal_cyclic", False):
        return TemporalFeatureEngineer()
    return "passthrough"


def build_pipeline(model_config: dict, preprocess_config: dict) -> Pipeline:
    return Pipeline(
        [
            ("features", build_feature_engineer(preprocess_config)),
            ("preprocess", build_preprocessor(preprocess_config)),
            ("classifier", XGBClassifier(**model_config)),
        ]
    )


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    """Metricas livres de threshold: ROC-AUC e PR-AUC (average precision).

    A PR-AUC e a metrica de referencia em bases desbalanceadas, pois resume a
    curva Precision x Recall sem depender de um ponto de corte arbitario.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Media e desvio-padrao entre dobras para cada metrica observada."""
    summary = {}
    for metric_name in fold_metrics[0]:
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def confusion_matrix_counts(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred
    ).ravel()
    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def select_business_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    evaluation_config: dict,
) -> dict:
    thresholds = np.linspace(
        evaluation_config["threshold_search_start"],
        evaluation_config["threshold_search_end"],
        evaluation_config["threshold_search_steps"],
    )

    target_precision = evaluation_config["target_precision"]
    target_recall = evaluation_config["target_recall"]
    precision_weight = evaluation_config["precision_weight"]
    recall_weight = evaluation_config["recall_weight"]

    best_result = None

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_classification_metrics(y_true, predictions)

        precision_gap = abs(metrics["precision"] - target_precision)
        recall_gap = abs(metrics["recall"] - target_recall)
        weighted_gap = (precision_gap * precision_weight) + (recall_gap * recall_weight)

        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
            "business_alignment": {
                "target_precision": float(target_precision),
                "target_recall": float(target_recall),
                "precision_gap": float(precision_gap),
                "recall_gap": float(recall_gap),
                "weighted_gap": float(weighted_gap),
                "alignment_score": float(max(0.0, 1.0 - weighted_gap)),
            },
        }

        if best_result is None:
            best_result = candidate
            continue

        current_gap = candidate["business_alignment"]["weighted_gap"]
        best_gap = best_result["business_alignment"]["weighted_gap"]
        current_metrics = candidate["metrics"]
        best_metrics = best_result["metrics"]

        if (
            current_gap < best_gap
            or (
                np.isclose(current_gap, best_gap)
                and current_metrics["recall"] > best_metrics["recall"]
            )
            or (
                np.isclose(current_gap, best_gap)
                and np.isclose(current_metrics["recall"], best_metrics["recall"])
                and current_metrics["precision"] > best_metrics["precision"]
            )
            or (
                np.isclose(current_gap, best_gap)
                and np.isclose(current_metrics["recall"], best_metrics["recall"])
                and np.isclose(current_metrics["precision"], best_metrics["precision"])
                and current_metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_result = candidate

    return {
        "selection_rule": "minimize_weighted_gap_to_business_targets",
        "search_space": {
            "start": float(evaluation_config["threshold_search_start"]),
            "end": float(evaluation_config["threshold_search_end"]),
            "steps": int(evaluation_config["threshold_search_steps"]),
            "precision_weight": float(precision_weight),
            "recall_weight": float(recall_weight),
        },
        "selected": best_result,
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    evaluation_config: dict,
    model_config: dict,
    preprocess_config: dict,
    random_state: int,
) -> tuple[np.ndarray, list]:
    """Executa o StratifiedKFold e devolve as probabilidades out-of-fold.

    Cada fold ajusta um Pipeline independente (feature engineering + preprocess +
    classificador) apenas sobre o seu conjunto de treino, garantindo que nenhuma
    estatistica vaze do treino para o teste.
    """
    splitter = StratifiedKFold(
        n_splits=evaluation_config["cv_folds"],
        shuffle=evaluation_config["shuffle"],
        random_state=random_state,
    )
    splits = list(splitter.split(X, y))

    oof_probabilities = np.zeros(len(y), dtype=float)
    for train_index, test_index in splits:
        pipeline = build_pipeline(model_config, preprocess_config)
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        oof_probabilities[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]

    return oof_probabilities, splits


def evaluate_seed_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    evaluation_config: dict,
    model_config: dict,
    preprocess_config: dict,
    primary_threshold: float,
    primary_summary: dict,
) -> dict:
    """Repete a validacao cruzada com outras seeds (repeated stratified CV).

    Objetivo: provar que o desempenho nao e um artefato da particao escolhida.
    Cada seed seleciona o seu proprio threshold operacional e reportamos a
    media e o desvio-padrao de precision/recall entre as seeds.
    """
    primary_seed = int(evaluation_config["random_state"])
    seeds = [int(seed) for seed in evaluation_config.get("robustness_seeds", [primary_seed])]

    per_seed = {}
    for seed in seeds:
        if seed == primary_seed:
            per_seed[seed] = {
                "threshold": float(primary_threshold),
                "precision": primary_summary["precision"]["mean"],
                "recall": primary_summary["recall"]["mean"],
            }
            continue

        seed_config = {**evaluation_config, "random_state": seed}
        oof_probabilities, splits = run_cross_validation(
            X, y, seed_config, model_config, preprocess_config, seed
        )
        seed_calibration = select_business_threshold(y, oof_probabilities, seed_config)
        seed_threshold = float(seed_calibration["selected"]["threshold"])
        fold_metrics = [
            compute_classification_metrics(
                y.iloc[test_index],
                (oof_probabilities[test_index] >= seed_threshold).astype(int),
            )
            for _, test_index in splits
        ]
        per_seed[seed] = {
            "threshold": seed_threshold,
            "precision": float(np.mean([fold["precision"] for fold in fold_metrics])),
            "recall": float(np.mean([fold["recall"] for fold in fold_metrics])),
        }

    precisions = np.array([entry["precision"] for entry in per_seed.values()], dtype=float)
    recalls = np.array([entry["recall"] for entry in per_seed.values()], dtype=float)
    return {
        "seeds": seeds,
        "per_seed": {str(seed): entry for seed, entry in per_seed.items()},
        "precision_mean": float(precisions.mean()),
        "precision_std": float(precisions.std(ddof=0)),
        "recall_mean": float(recalls.mean()),
        "recall_std": float(recalls.std(ddof=0)),
    }


def evaluate_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    evaluation_config: dict,
    model_config: dict,
    preprocess_config: dict,
) -> tuple[dict, np.ndarray]:
    oof_probabilities, splits = run_cross_validation(
        X,
        y,
        evaluation_config,
        model_config,
        preprocess_config,
        random_state=int(evaluation_config["random_state"]),
    )

    threshold_calibration = select_business_threshold(y, oof_probabilities, evaluation_config)
    operating_threshold = float(threshold_calibration["selected"]["threshold"])

    # Metricas por dobra avaliadas NO THRESHOLD OPERACIONAL (nao em 0.5), pois e
    # nessa faixa que o modelo efetivamente opera. ROC-AUC/PR-AUC entram como
    # metricas livres de threshold, calculadas sobre as probabilidades do fold.
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
    cross_validation_mean_metrics = {
        metric_name: values["mean"] for metric_name, values in fold_summary.items()
    }

    operating_predictions = (oof_probabilities >= operating_threshold).astype(int)
    metrics = {
        "cv_strategy": "StratifiedKFold",
        "operating_threshold": operating_threshold,
        "cross_validation_mean_metrics": cross_validation_mean_metrics,
        "cross_validation_fold_metrics": fold_metrics,
        "cross_validation_summary": fold_summary,
        "out_of_fold_default_threshold_metrics": compute_classification_metrics(
            y, (oof_probabilities >= DEFAULT_THRESHOLD).astype(int)
        ),
        "out_of_fold_operating_threshold_metrics": compute_classification_metrics(
            y, operating_predictions
        ),
        "out_of_fold_probability_metrics": compute_probability_metrics(y, oof_probabilities),
        "out_of_fold_confusion_matrix": confusion_matrix_counts(y, operating_predictions),
        "threshold_calibration": threshold_calibration,
        "seed_robustness": evaluate_seed_robustness(
            X,
            y,
            evaluation_config,
            model_config,
            preprocess_config,
            operating_threshold,
            fold_summary,
        ),
    }
    return metrics, oof_probabilities


def evaluate_model_acceptance(metrics: dict, acceptance_config: dict) -> dict:
    checks = {
        "min_cv_precision": metrics["cross_validation_mean_metrics"]["precision"]
        >= acceptance_config["min_cv_precision"],
        "min_cv_recall": metrics["cross_validation_mean_metrics"]["recall"] >= acceptance_config["min_cv_recall"],
        "min_calibrated_precision": metrics["threshold_calibration"]["selected"]["metrics"]["precision"]
        >= acceptance_config["min_calibrated_precision"],
        "min_calibrated_recall": metrics["threshold_calibration"]["selected"]["metrics"]["recall"]
        >= acceptance_config["min_calibrated_recall"],
        "max_business_weighted_gap": metrics["threshold_calibration"]["selected"]["business_alignment"]["weighted_gap"]
        <= acceptance_config["max_business_weighted_gap"],
    }
    return {
        "criteria": {key: float(value) for key, value in acceptance_config.items()},
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def train_final_model(X: pd.DataFrame, y: pd.Series, model_config: dict, preprocess_config: dict) -> None:
    pipeline = build_pipeline(model_config, preprocess_config)
    pipeline.fit(X, y)
    PIPELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_MODEL_PATH)


def describe_pipeline(preprocess_config: dict) -> str:
    feature_config = preprocess_config.get("feature_engineering", {}) or {}
    steps = []
    if feature_config.get("temporal_cyclic", False):
        steps.append("TemporalFeatureEngineer(hour,hour_sin,hour_cos)")
    steps.append("ColumnTransformer(PowerTransformer+MinMaxScaler)")
    steps.append("XGBClassifier")
    return " + ".join(steps)


def build_evaluation_report(
    metrics: dict,
    evaluation_config: dict,
    preprocess_config: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Consolida a avaliacao auditavel: por dobra, resumo, out-of-fold e robustez."""
    total_rows = int(len(y))
    positive_count = int(y.sum())
    return {
        "generated_by": "src/config/train_model.py",
        "validation": {
            "cv_strategy": metrics["cv_strategy"],
            "cv_folds": int(evaluation_config["cv_folds"]),
            "shuffle": bool(evaluation_config["shuffle"]),
            "random_state": int(evaluation_config["random_state"]),
            "operating_threshold": float(metrics["operating_threshold"]),
        },
        "dataset": {
            "processed_data_path": relative_posix_path(PROCESSED_DATA_PATH),
            "row_count": total_rows,
            "feature_count": int(X.shape[1]),
            "positive_count": positive_count,
            "positive_rate": float(positive_count / total_rows),
            "preprocess": preprocess_config,
        },
        "cross_validation": {
            "fold_metrics": metrics["cross_validation_fold_metrics"],
            "summary": metrics["cross_validation_summary"],
        },
        "out_of_fold": {
            "default_threshold_metrics": metrics["out_of_fold_default_threshold_metrics"],
            "operating_threshold_metrics": metrics["out_of_fold_operating_threshold_metrics"],
            "probability_metrics": metrics["out_of_fold_probability_metrics"],
            "confusion_matrix": metrics["out_of_fold_confusion_matrix"],
        },
        "threshold_calibration": metrics["threshold_calibration"],
        "seed_robustness": metrics["seed_robustness"],
        "evidence": {
            "precision_recall_curve": relative_posix_path(PR_CURVE_PATH),
            "confusion_matrix": relative_posix_path(CONFUSION_MATRIX_PATH),
        },
    }


def render_evaluation_markdown(report: dict) -> str:
    """Renderiza o relatorio de avaliacao em Markdown legivel para auditoria."""
    validation = report["validation"]
    dataset = report["dataset"]
    summary = report["cross_validation"]["summary"]
    out_of_fold = report["out_of_fold"]
    matrix = out_of_fold["confusion_matrix"]
    robustness = report["seed_robustness"]

    lines = [
        "# Relatorio de Avaliacao — Deteccao de Fraude (Credicard)",
        "",
        "> Gerado automaticamente por `src/config/train_model.py`. Nao editar manualmente.",
        "",
        "## Estrategia de validacao",
        f"- {validation['cv_strategy']} com {validation['cv_folds']} dobras "
        f"(shuffle={validation['shuffle']}, seed={validation['random_state']})",
        f"- Threshold operacional selecionado: {validation['operating_threshold']:.3f}",
        f"- Dataset: {dataset['row_count']} linhas | {dataset['feature_count']} features | "
        f"{dataset['positive_count']} fraudes ({dataset['positive_rate']:.4%})",
        "",
        "## Metricas por dobra (threshold operacional)",
        "| dobra | " + " | ".join(FOLD_METRIC_ORDER) + " |",
        "| " + " | ".join(["---"] * (len(FOLD_METRIC_ORDER) + 1)) + " |",
    ]
    for index, fold in enumerate(report["cross_validation"]["fold_metrics"], start=1):
        formatted = " | ".join(f"{fold[name]:.4f}" for name in FOLD_METRIC_ORDER)
        lines.append(f"| {index} | {formatted} |")

    lines += [
        "",
        "## Resumo entre dobras (media +/- desvio-padrao)",
        "| metrica | media | desvio-padrao |",
        "| --- | --- | --- |",
    ]
    for metric_name, values in summary.items():
        lines.append(f"| {metric_name} | {values['mean']:.4f} | {values['std']:.4f} |")

    operating = out_of_fold["operating_threshold_metrics"]
    default = out_of_fold["default_threshold_metrics"]
    probability = out_of_fold["probability_metrics"]
    default_line = (
        f"- Threshold 0,500: precision {default['precision']:.4f} | "
        f"recall {default['recall']:.4f} | f1 {default['f1']:.4f}"
    )
    operating_line = (
        f"- Threshold {validation['operating_threshold']:.3f}: "
        f"precision {operating['precision']:.4f} | recall {operating['recall']:.4f} | "
        f"f1 {operating['f1']:.4f}"
    )
    probability_line = (
        f"- Livres de threshold: ROC-AUC {probability['roc_auc']:.4f} | "
        f"PR-AUC {probability['pr_auc']:.4f}"
    )
    lines += [
        "",
        "## Metricas out-of-fold (agregadas)",
        default_line,
        operating_line,
        probability_line,
        "",
        "## Matriz de confusao (threshold operacional)",
        "| Real \\ Predito | Legitimo | Fraude |",
        "| --- | --- | --- |",
        f"| Legitimo | {matrix['true_negative']} | {matrix['false_positive']} |",
        f"| Fraude | {matrix['false_negative']} | {matrix['true_positive']} |",
        "",
        "## Robustez a seed (repeated stratified CV)",
        "| seed | threshold | precision | recall |",
        "| --- | --- | --- | --- |",
    ]
    for seed, entry in robustness["per_seed"].items():
        lines.append(
            f"| {seed} | {entry['threshold']:.3f} | "
            f"{entry['precision']:.4f} | {entry['recall']:.4f} |"
        )
    lines += [
        (
            f"| **media** | | **{robustness['precision_mean']:.4f}** | "
            f"**{robustness['recall_mean']:.4f}** |"
        ),
        (
            f"| **desvio-padrao** | | {robustness['precision_std']:.4f} | "
            f"{robustness['recall_std']:.4f} |"
        ),
        "",
        "## Evidencias graficas",
        f"- Curva Precision-Recall: `{report['evidence']['precision_recall_curve']}`",
        f"- Matriz de confusao: `{report['evidence']['confusion_matrix']}`",
        "",
    ]
    return "\n".join(lines)


def save_evaluation_report(report: dict) -> None:
    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVALUATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with EVALUATION_REPORT_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(render_evaluation_markdown(report))


def save_evaluation_plots(
    y_true: pd.Series,
    probabilities: np.ndarray,
    operating_threshold: float,
) -> None:
    """Versiona evidencias graficas: curva Precision-Recall e matriz de confusao."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    EVALUATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= operating_threshold).astype(int)

    precision_values, recall_values, _ = precision_recall_curve(y_true, probabilities)
    average_precision = float(average_precision_score(y_true, probabilities))

    pr_figure, pr_axis = plt.subplots(figsize=(6.5, 5.0))
    pr_axis.plot(recall_values, precision_values, color="#1f77b4", label="Curva Precision-Recall")
    pr_axis.scatter(
        [recall_score(y_true, predictions, zero_division=0)],
        [precision_score(y_true, predictions, zero_division=0)],
        color="#d62728",
        zorder=5,
        label=f"Threshold operacional = {operating_threshold:.3f}",
    )
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title(f"Curva Precision-Recall (PR-AUC = {average_precision:.4f})")
    pr_axis.set_xlim(0.0, 1.0)
    pr_axis.set_ylim(0.0, 1.05)
    pr_axis.grid(alpha=0.3)
    pr_axis.legend(loc="lower left")
    pr_figure.tight_layout()
    pr_figure.savefig(PR_CURVE_PATH, dpi=120)
    plt.close(pr_figure)

    matrix = confusion_matrix(y_true, predictions)
    cm_figure, cm_axis = plt.subplots(figsize=(6.0, 5.0))
    cm_axis.imshow(matrix, cmap="Blues")
    highlight_limit = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            cm_axis.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > highlight_limit else "black",
            )
    cm_axis.set_xticks([0, 1], labels=["Legitimo", "Fraude"])
    cm_axis.set_yticks([0, 1], labels=["Legitimo", "Fraude"])
    cm_axis.set_xlabel("Predito")
    cm_axis.set_ylabel("Real")
    cm_axis.set_title(f"Matriz de confusao @ threshold {operating_threshold:.3f}")
    cm_figure.tight_layout()
    cm_figure.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close(cm_figure)


def save_metrics(
    metrics: dict,
    evaluation_config: dict,
    acceptance_config: dict,
    model_config: dict,
    preprocess_config: dict,
) -> None:
    oof_default_metrics = metrics["out_of_fold_default_threshold_metrics"]
    oof_operating_metrics = metrics["out_of_fold_operating_threshold_metrics"]
    oof_probability_metrics = metrics["out_of_fold_probability_metrics"]

    metrics_payload = {
        "dataset": {
            "processed_data_path": str(PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)),
        },
        "evaluation": {
            "cv_folds": int(evaluation_config["cv_folds"]),
            "cv_shuffle": bool(evaluation_config["shuffle"]),
            "cv_random_state": int(evaluation_config["random_state"]),
            "default_threshold": DEFAULT_THRESHOLD,
            "cv_strategy": metrics["cv_strategy"],
            "operating_threshold": metrics["operating_threshold"],
            "cross_validation_mean_metrics": metrics["cross_validation_mean_metrics"],
            "out_of_fold_default_threshold_metrics": oof_default_metrics,
            "out_of_fold_operating_threshold_metrics": oof_operating_metrics,
            "out_of_fold_probability_metrics": oof_probability_metrics,
            "threshold_calibration": metrics["threshold_calibration"],
        },
        "business_target": {
            "precision": float(evaluation_config["target_precision"]),
            "recall": float(evaluation_config["target_recall"]),
        },
        "acceptance": evaluate_model_acceptance(metrics, acceptance_config),
        "model": {
            "artifact_path": str(PIPELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "type": "XGBClassifier",
            "pipeline": describe_pipeline(preprocess_config),
            "random_state": int(model_config["random_state"]),
            "max_depth": int(model_config["max_depth"]),
            "n_estimators": int(model_config["n_estimators"]),
            "learning_rate": float(model_config["learning_rate"]),
            "scale_pos_weight": float(model_config["scale_pos_weight"]),
        },
    }
    with (PROJECT_ROOT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)


def save_model_metadata(metrics: dict) -> None:
    selected_threshold = metrics["evaluation"]["threshold_calibration"]["selected"]
    metadata = {
        "model_artifact": metrics["model"]["artifact_path"],
        "decision_threshold": selected_threshold["threshold"],
        "out_of_fold_calibrated_metrics": selected_threshold["metrics"],
        "business_alignment": selected_threshold["business_alignment"],
        "acceptance": metrics["acceptance"],
    }
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    evaluation_config, acceptance_config, base_model_config, preprocess_config = load_training_config()
    model_config = build_model_config(base_model_config, y)

    metrics, oof_probabilities = evaluate_with_cross_validation(
        X, y, evaluation_config, model_config, preprocess_config
    )
    train_final_model(X, y, model_config, preprocess_config)
    save_metrics(metrics, evaluation_config, acceptance_config, model_config, preprocess_config)

    evaluation_report = build_evaluation_report(metrics, evaluation_config, preprocess_config, X, y)
    save_evaluation_report(evaluation_report)
    save_evaluation_plots(y, oof_probabilities, float(metrics["operating_threshold"]))

    save_model_metadata(
        {
            "evaluation": metrics,
            "acceptance": evaluate_model_acceptance(metrics, acceptance_config),
            "model": {
                "artifact_path": str(PIPELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
            },
        }
    )

    selected_threshold = metrics["threshold_calibration"]["selected"]
    acceptance = evaluate_model_acceptance(metrics, acceptance_config)
    if not acceptance["passed"]:
        raise RuntimeError(f"Modelo reprovado nos criterios de aceite: {acceptance['checks']}")

    print(
        "Modelo treinado com validacao cruzada estratificada (Pipeline anti-leakage). "
        f"Threshold operacional: {selected_threshold['threshold']:.3f} | "
        f"Precisao: {selected_threshold['metrics']['precision']:.4f} | "
        f"Recall: {selected_threshold['metrics']['recall']:.4f} | "
        f"Aceite: {'PASS' if acceptance['passed'] else 'FAIL'}"
    )
    print(f"Relatorio de avaliacao salvo em {EVALUATION_REPORT_PATH} e {EVALUATION_REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
