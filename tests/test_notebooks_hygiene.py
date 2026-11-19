"""Guarda de higiene dos notebooks (Fase 9: limpeza e determinismo).

Os notebooks em `notebooks/` sao artefatos historicos versionados. Estes testes
garantem que eles permanecem limpos (sem saidas/execucao) e nao reintroduzem
referencias quebradas ao projeto antigo.
"""

import json

import pytest

from src.config.paths import PROJECT_ROOT

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATHS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

REMOVED_CONSTANTS = ("DADOS_CREDICARD_TRATADO", "DADOS_CREDICARD")
STALE_PROJECT_PATH = "Projeto_4_Modelo_de_Identificacao_de_Fraude"


def load_notebook(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def notebook_source(data):
    return "\n".join("".join(cell.get("source", [])) for cell in data["cells"])


@pytest.mark.parametrize("caminho", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebooks_have_no_stored_outputs_or_execution_counts(caminho):
    data = load_notebook(caminho)
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("outputs") == [], f"{caminho.name}: celula com saidas armazenadas"
        assert cell.get("execution_count") is None, (
            f"{caminho.name}: celula com execution_count preenchido"
        )


@pytest.mark.parametrize("caminho", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebooks_do_not_reference_removed_constants(caminho):
    source = notebook_source(load_notebook(caminho))
    for constant in REMOVED_CONSTANTS:
        assert constant not in source, f"{caminho.name}: usa constante removida {constant}"


@pytest.mark.parametrize("caminho", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebooks_do_not_reference_stale_project_path(caminho):
    source = notebook_source(load_notebook(caminho))
    assert STALE_PROJECT_PATH not in source, f"{caminho.name}: referencia o projeto antigo"
