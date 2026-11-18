import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from xgboost import XGBClassifier

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.feature_engineering import TemporalFeatureEngineer
from src.config.paths import MODEL_METADATA_PATH, PIPELINE_MODEL_PATH, PROCESSED_DATA_PATH, PROJECT_ROOT

DEFAULT_THRESHOLD = 0.5


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


def evaluate_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    evaluation_config: dict,
    model_config: dict,
    preprocess_config: dict,
) -> dict:
    splitter = StratifiedKFold(
        n_splits=evaluation_config["cv_folds"],
        shuffle=evaluation_config["shuffle"],
        random_state=evaluation_config["random_state"],
    )
    splits = list(splitter.split(X, y))

    oof_probabilities = np.zeros(len(y), dtype=float)
    for train_index, test_index in splits:
        pipeline = build_pipeline(model_config, preprocess_config)
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        oof_probabilities[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]

    threshold_calibration = select_business_threshold(y, oof_probabilities, evaluation_config)
    operating_threshold = float(threshold_calibration["selected"]["threshold"])

    # Metricas do gate de negocio avaliadas NO THRESHOLD OPERACIONAL (nao em 0.5),
    # pois e nessa faixa que o modelo efetivamente opera em producao.
    fold_metrics = [
        compute_classification_metrics(
            y.iloc[test_index],
            (oof_probabilities[test_index] >= operating_threshold).astype(int),
        )
        for _, test_index in splits
    ]

    cross_validation_mean_metrics = {
        "accuracy": float(np.mean([fold["accuracy"] for fold in fold_metrics])),
        "precision": float(np.mean([fold["precision"] for fold in fold_metrics])),
        "recall": float(np.mean([fold["recall"] for fold in fold_metrics])),
    }

    return {
        "cv_strategy": "StratifiedKFold",
        "operating_threshold": operating_threshold,
        "cross_validation_mean_metrics": cross_validation_mean_metrics,
        "out_of_fold_default_threshold_metrics": compute_classification_metrics(
            y, (oof_probabilities >= DEFAULT_THRESHOLD).astype(int)
        ),
        "out_of_fold_operating_threshold_metrics": compute_classification_metrics(
            y, (oof_probabilities >= operating_threshold).astype(int)
        ),
        "threshold_calibration": threshold_calibration,
    }


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


def save_metrics(
    metrics: dict,
    evaluation_config: dict,
    acceptance_config: dict,
    model_config: dict,
    preprocess_config: dict,
) -> None:
    metrics_payload = {
        "dataset": {
            "processed_data_path": str(PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)),
        },
        "evaluation": {
            "cv_folds": int(evaluation_config["cv_folds"]),
            "cv_shuffle": bool(evaluation_config["shuffle"]),
            "cv_random_state": int(evaluation_config["random_state"]),
            "default_threshold": DEFAULT_THRESHOLD,
            **metrics,
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

    metrics = evaluate_with_cross_validation(X, y, evaluation_config, model_config, preprocess_config)
    train_final_model(X, y, model_config, preprocess_config)
    save_metrics(metrics, evaluation_config, acceptance_config, model_config, preprocess_config)
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


if __name__ == "__main__":
    main()
