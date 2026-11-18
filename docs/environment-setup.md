# Setup de Ambiente

## Recomendação

O ambiente reprodutível é gerenciado pelo **uv** a partir do [pyproject.toml](../pyproject.toml),
do [uv.lock](../uv.lock) e do [.python-version](../.python-version). Isso elimina warnings de
ambientes legados e garante as mesmas versões na sua máquina e na CI.

## Instalação

```powershell
# 1. Instale o uv (uma vez): https://docs.astral.sh/uv/
# 2. Na raiz do projeto, sincronize o ambiente (cria .venv e instala tudo)
uv sync

# 3. Valide a árvore de dependências
uv pip check
```

## Verificações recomendadas

```powershell
uv run pytest -q
uv run dvc pull
uv run python -m src.serving.cli --input dados/credicard_tratado.parquet --output predictions.csv --limit 5 --format csv
```

## Alternativa com pip

O [requirements.txt](../requirements.txt) é mantido como espelho dos pins diretos:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pip check
```

## Notas de ambiente

- Prefira `uv run <comando>` para garantir que a execução use o interpretador da `.venv`.
- Use `uv pip check` para detectar conflitos instalados no ambiente.
- O `numexpr` vem fixado no `uv.lock`; ambientes manuais antigos podem manter uma
  versão anterior e continuar emitindo warning.
- O warning de `fpdf` aparece quando `PyFPDF` e `fpdf2` coexistem no mesmo ambiente global.
- A suíte `pytest` filtra apenas warnings pendentes conhecidos de bibliotecas terceiras
  (`shap`) para manter a saída de validação objetiva.
