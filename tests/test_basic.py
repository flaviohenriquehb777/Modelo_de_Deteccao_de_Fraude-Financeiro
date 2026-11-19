import importlib
import sys

import pytest

# Dependencias de runtime cuja presenca no ambiente precisa ser garantida.
DEPENDENCIAS = (
    "dvc",
    "joblib",
    "mlflow",
    "numpy",
    "pandas",
    "pyarrow",
    "shap",
    "sklearn",
    "xgboost",
    "fpdf",
)


def test_imports():
    """Testa se todas as dependências podem ser importadas"""
    try:
        for modulo in DEPENDENCIAS:
            importlib.import_module(modulo)
        assert hasattr(importlib.import_module("fpdf"), "FPDF")
    except ImportError as e:
        pytest.fail(f"❌ Falha ao importar: {str(e)}")


def test_python_version():
    """Testa se estamos usando Python 3.7+"""
    assert sys.version_info >= (3, 7), "Python version must be 3.7 or higher"
