import json
import sys
from pathlib import Path

import yaml

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import (
    HISTORICAL_MODEL_REFERENCE_PATH,
    MLFLOW_PUBLIC_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
)


DAGSHUB_BASE_URL = "https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro"

# Snapshot do registry publico do DagsHub, VERIFICADO contra a API MLflow publica
# (https://dagshub.com/<owner>/<repo>.mlflow/api/2.0/mlflow/...). O `mlruns/` local
# e git-ignored e nao carrega o historico legado, portanto os fatos que existem
# apenas no registry publico (nome/id do experimento, contagens e modelo historico)
# sao DECLARADOS aqui a partir dessa verificacao. Nenhum run_id e inventado: o
# detalhe por-run do resumo vem exclusivamente do `mlruns/` local (train_model.py --mlflow).
DAGSHUB_EXPERIMENT_ID = "0"
DAGSHUB_EXPERIMENT_NAME = "Fraud Detection XGBoost Experiment"
DAGSHUB_PUBLIC_RUNS = 7
DAGSHUB_PUBLIC_MODELS = 1
DAGSHUB_BEST_MODEL_ID = "m-695031f059db422fb7c12c7e47ef58e9"
DAGSHUB_BEST_MODEL_NAME = "Treinamento do Modelo - XGBoost Otimizado"

LOCAL_RUN_METRICS = (
    "operating_threshold",
    "calibrated_precision",
    "calibrated_recall",
    "calibrated_f1",
    "cv_precision",
    "cv_recall",
    "pr_auc",
    "roc_auc",
    "business_weighted_gap",
    "acceptance_passed",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_metric(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    line = metric_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    parts = line.split()
    return float(parts[1]) if len(parts) >= 2 else None


def collect_local_runs(mlruns_dir: Path) -> list[dict]:
    """Le as execucoes REAIS registradas em `mlruns/` (via `train_model.py --mlflow`).

    Retorna lista vazia se o diretorio nao existir (ex.: CI limpa), garantindo que
    o resumo nunca dependa de run_ids fabricados.
    """
    if not mlruns_dir.exists():
        return []

    runs = []
    for experiment_dir in sorted(mlruns_dir.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name == ".trash":
            continue
        if not (experiment_dir / "meta.yaml").exists():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "models":
                continue
            meta_path = run_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            meta = read_yaml(meta_path)
            entry = {
                "run_id": meta["run_id"],
                "run_name": meta.get("run_name"),
                "experiment_id": meta.get("experiment_id", experiment_dir.name),
                "status": meta.get("status"),
                "created_at": meta.get("start_time"),
            }
            for metric_name in LOCAL_RUN_METRICS:
                entry[metric_name] = read_metric(run_dir / "metrics" / metric_name)
            runs.append(entry)
    return sorted(runs, key=lambda run: run["created_at"] or 0)


def build_historical_best_model(reference: dict) -> dict:
    baseline = reference["business_baseline"]
    return {
        "model_id": DAGSHUB_BEST_MODEL_ID,
        "name": DAGSHUB_BEST_MODEL_NAME,
        "mean_precision": baseline["precision"],
        "mean_recall": baseline["recall"],
        "reference_artifact": "models/best_model_xgboost.reference.json",
        "selection_rule": "highest_mean_precision_then_mean_recall_then_mean_accuracy",
        "business_baseline_alignment": {
            "target_precision": baseline["precision"],
            "target_recall": baseline["recall"],
        },
    }


def build_operational_model(metrics: dict, model_metadata: dict) -> dict:
    """Modelo operacional atual, alinhado a `metrics.json` (fonte de verdade)."""
    model = metrics["model"]
    calibrated = model_metadata["out_of_fold_calibrated_metrics"]
    return {
        "source": "metrics.json",
        "artifact_path": model["artifact_path"],
        "type": model["type"],
        "pipeline": model["pipeline"],
        "operating_threshold": metrics["evaluation"]["operating_threshold"],
        "hyperparameters": {
            "max_depth": model["max_depth"],
            "n_estimators": model["n_estimators"],
            "learning_rate": model["learning_rate"],
            "random_state": model["random_state"],
            "scale_pos_weight": model["scale_pos_weight"],
        },
        "calibrated_precision": calibrated["precision"],
        "calibrated_recall": calibrated["recall"],
        "acceptance_passed": metrics["acceptance"]["passed"],
    }


def main() -> None:
    metrics = read_json(PROJECT_ROOT / "metrics.json")
    model_metadata = read_json(MODEL_METADATA_PATH)
    reference = read_json(HISTORICAL_MODEL_REFERENCE_PATH)
    local_runs = collect_local_runs(PROJECT_ROOT / "mlruns")

    summary = {
        "dagshub": {
            "repository_url": DAGSHUB_BASE_URL,
            "experiments_url": f"{DAGSHUB_BASE_URL}/experiments",
            "models_url": f"{DAGSHUB_BASE_URL}/models",
            "repository_visibility": "public",
        },
        "mlflow_experiment": {
            "experiment_id": DAGSHUB_EXPERIMENT_ID,
            "name": DAGSHUB_EXPERIMENT_NAME,
            "total_public_runs": DAGSHUB_PUBLIC_RUNS,
            "total_public_models": DAGSHUB_PUBLIC_MODELS,
            "public_snapshot_note": (
                "Contagens verificadas na API MLflow publica do DagsHub. O detalhe "
                "por-run verificavel fica em `local_mlflow_runs`."
            ),
        },
        "historical_best_model": build_historical_best_model(reference),
        "operational_model": build_operational_model(metrics, model_metadata),
        "local_mlflow_runs": local_runs,
    }

    MLFLOW_PUBLIC_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MLFLOW_PUBLIC_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Resumo publico de MLflow/DagsHub salvo em {MLFLOW_PUBLIC_SUMMARY_PATH} "
        f"({len(local_runs)} run(s) local(is) real(is))."
    )


if __name__ == "__main__":
    main()
