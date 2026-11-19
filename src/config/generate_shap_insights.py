import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import (
    MODEL_METADATA_PATH,
    PIPELINE_MODEL_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    SHAP_EXPLANATIONS_DIR,
    SHAP_MANIFEST_PATH,
)

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)
sns.set_palette("viridis")


def load_explainability_config() -> dict:
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params["explainability"]


def build_analysis_sample(data: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(data) <= sample_size:
        return data.copy()

    target_column = "Class"
    positive = data[data[target_column] == 1]
    negative = data[data[target_column] == 0]

    positive_sample_size = min(len(positive), max(1, sample_size // 2))
    negative_sample_size = min(len(negative), sample_size - positive_sample_size)

    sampled = pd.concat(
        [
            positive.sample(n=positive_sample_size, random_state=random_state),
            negative.sample(n=negative_sample_size, random_state=random_state),
        ]
    ).sample(frac=1.0, random_state=random_state)

    return sampled.reset_index(drop=True)


class FraudSHAPExplainer:
    def __init__(self, model_path: Path, data_path: Path, output_dir: Path, explainability_config: dict):
        self.pipeline = joblib.load(model_path)
        self.data = pd.read_parquet(data_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainability_config = explainability_config

        sampled_data = build_analysis_sample(
            self.data,
            explainability_config["sample_size"],
            explainability_config["random_state"],
        )
        raw_features = sampled_data.drop("Class", axis=1)
        self.y = sampled_data["Class"]

        self.feature_engineer = self._resolve_step("features")
        self.preprocessor = self._resolve_step("preprocess")
        self.classifier = self._resolve_classifier()

        # O artefato de producao e um Pipeline que recebe features BRUTAS. Para o
        # SHAP precisamos explicar o classificador, entao aplicamos as etapas do
        # proprio Pipeline (engenharia de features + pre-processamento, ajustadas
        # no treino) na mesma ordem em que o modelo as aplica.
        engineered_features = raw_features
        if self.feature_engineer is not None:
            engineered_features = self.feature_engineer.transform(raw_features)

        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(engineered_features)
            feature_names = list(self.preprocessor.get_feature_names_out())
            self.X = pd.DataFrame(transformed, columns=feature_names, index=raw_features.index)
        else:
            self.X = engineered_features

        print(f"Dados carregados para explainability: {self.X.shape}")
        print(f"Classificador explicado: {type(self.classifier).__name__}")

    def _resolve_step(self, step_name: str):
        steps = getattr(self.pipeline, "named_steps", None)
        if steps and step_name in steps:
            return steps[step_name]
        return None

    def _resolve_classifier(self):
        steps = getattr(self.pipeline, "named_steps", None)
        if steps and "classifier" in steps:
            return self._unwrap_calibrated(steps["classifier"])
        return self.pipeline

    @staticmethod
    def _unwrap_calibrated(estimator):
        """Extrai o estimador base quando o passo final e um CalibratedClassifierCV.

        A calibracao apenas reescala as probabilidades de saida; o SHAP explica o
        modelo subjacente (arvore), portanto devolvemos o estimador interno.
        """
        if isinstance(estimator, CalibratedClassifierCV):
            calibrated = estimator.calibrated_classifiers_[0]
            if hasattr(calibrated, "estimator"):
                return calibrated.estimator
            return getattr(calibrated, "base_estimator", calibrated)
        return estimator

    def _seed_randomness(self) -> None:
        """Fixa as sementes dos geradores usados internamente pelo SHAP.

        Os plots beeswarm (`summary_plot`), de dependencia e o `force_plot` usam
        `np.random` e `random` (jitter, embaralhamento e ids do HTML). Sem fixar
        as sementes os PNGs/HTML mudam a cada execucao, quebrando a idempotencia
        do pipeline DVC. Usamos o `random_state` da configuracao de explainability.
        """
        random_state = self.explainability_config["random_state"]
        np.random.seed(random_state)
        random.seed(random_state)

    def generate_shap_analysis(self) -> None:
        print("Gerando analise SHAP...")
        self._seed_randomness()

        self.explainer = shap.TreeExplainer(self.classifier)
        shap_values = self.explainer.shap_values(self.X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        self._plot_summary(shap_values)
        importance_df = self._plot_feature_importance(shap_values)
        self._generate_dependence_plots(shap_values, importance_df)
        self._generate_force_plot(shap_values)
        self._save_shap_values(shap_values, importance_df)
        self._save_manifest(importance_df)

        print(f"Analise SHAP salva em: {self.output_dir}")

    def _plot_summary(self, shap_values: np.ndarray) -> None:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": self.X.columns,
                "importance": np.abs(shap_values).mean(0),
            }
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=importance_df.head(15))
        plt.title("Top 15 Features por Importancia SHAP")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_feature_importance.png", dpi=300)
        plt.close()

        importance_df.to_csv(self.output_dir / "feature_importance_ranking.csv", index=False)
        return importance_df

    def _generate_dependence_plots(self, shap_values: np.ndarray, importance_df: pd.DataFrame) -> None:
        dependence_dir = self.output_dir / "dependence_plots"
        dependence_dir.mkdir(exist_ok=True)

        top_features = importance_df.head(self.explainability_config["top_features"])["feature"].tolist()
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, self.X, show=False)
            plt.title(f"Dependence Plot: {feature}")
            plt.tight_layout()
            plt.savefig(dependence_dir / f"dependence_{feature}.png", dpi=300)
            plt.close()

    def _generate_force_plot(self, shap_values: np.ndarray) -> None:
        fraud_rows = self.y[self.y == 1]
        if fraud_rows.empty:
            return

        instance_position = int(fraud_rows.index[0])
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        force_plot = shap.force_plot(
            expected_value,
            shap_values[instance_position],
            self.X.iloc[instance_position],
            show=False,
        )
        shap.save_html(str(self.output_dir / "shap_force_plot.html"), force_plot)

    def _save_shap_values(self, shap_values: np.ndarray, importance_df: pd.DataFrame) -> None:
        shap_df = pd.DataFrame(shap_values, columns=self.X.columns)
        shap_df.to_csv(self.output_dir / "shap_values.csv", index=False)

        top_features = []
        for feature_name in importance_df.head(10)["feature"].tolist():
            feature_position = self.X.columns.get_loc(feature_name)
            top_features.append(
                {
                    "feature": feature_name,
                    "importance": float(importance_df.loc[importance_df["feature"] == feature_name, "importance"].iloc[0]),
                    "direction": "positive" if float(np.mean(shap_values[:, feature_position])) > 0 else "negative",
                }
            )

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        stats = {
            "sample_size": int(len(self.X)),
            "total_features": int(len(self.X.columns)),
            "mean_expected_value": float(expected_value),
            "top_features": top_features,
        }
        with (self.output_dir / "shap_stats.json").open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)

    def _save_manifest(self, importance_df: pd.DataFrame) -> None:
        model_metadata = {}
        if MODEL_METADATA_PATH.exists():
            with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
                model_metadata = json.load(file)

        manifest = {
            "model_path": str(PIPELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "data_path": str(PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)),
            "sample_size": int(len(self.X)),
            "top_features": importance_df.head(self.explainability_config["top_features"])["feature"].tolist(),
            "decision_threshold": model_metadata.get("decision_threshold"),
            "generated_files": sorted(
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                for path in self.output_dir.rglob("*")
                if path.is_file()
            ),
        }
        with SHAP_MANIFEST_PATH.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)


def main() -> None:
    explainability_config = load_explainability_config()
    explainer = FraudSHAPExplainer(
        PIPELINE_MODEL_PATH,
        PROCESSED_DATA_PATH,
        SHAP_EXPLANATIONS_DIR,
        explainability_config,
    )
    explainer.generate_shap_analysis()

    stats_file = SHAP_EXPLANATIONS_DIR / "shap_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Arquivo esperado nao foi gerado: {stats_file}")

    with stats_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Top feature registrada: {data['top_features'][0]['feature']}")


if __name__ == "__main__":
    main()
