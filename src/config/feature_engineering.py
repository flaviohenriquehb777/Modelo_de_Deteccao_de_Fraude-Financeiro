"""Engenharia de features fold-safe para o Pipeline de deteccao de fraude.

As transformacoes aqui sao puramente row-wise (deterministicas): nao estimam
parametros a partir dos dados de treino. Isso as torna seguras para viver dentro
do Pipeline, pois nao introduzem vazamento de dados entre os folds e garantem
que treino e inferencia apliquem exatamente a mesma receita.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deriva features temporais ciclicas a partir da coluna `Time` (segundos).

    `Time` representa os segundos decorridos desde a primeira transacao do
    dataset. Convertemos esse valor na hora do dia (0-23) e a codificamos de
    forma ciclica (seno/cosseno), evitando a descontinuidade artificial entre
    23h e 0h que uma codificacao linear imporia.
    """

    def __init__(self, time_column: str = "Time", seconds_per_hour: int = 3600, hours_in_day: int = 24):
        self.time_column = time_column
        self.seconds_per_hour = seconds_per_hour
        self.hours_in_day = hours_in_day

    def fit(self, X: pd.DataFrame, y: object = None) -> "TemporalFeatureEngineer":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        hour = (transformed[self.time_column].astype(float) // self.seconds_per_hour) % self.hours_in_day
        angle = 2.0 * np.pi * hour / self.hours_in_day
        transformed["hour"] = hour.astype(float)
        transformed["hour_sin"] = np.sin(angle)
        transformed["hour_cos"] = np.cos(angle)
        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is not None:
            base = list(input_features)
        else:
            base = list(getattr(self, "feature_names_in_", []))
        return np.asarray([*base, "hour", "hour_sin", "hour_cos"], dtype=object)
