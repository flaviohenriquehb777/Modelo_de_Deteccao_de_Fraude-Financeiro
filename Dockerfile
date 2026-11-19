# =============================================================================
# Imagem de serving de REFERENCIA (nao ha deploy/producao).
# Prova que o modelo empacota e sobe como servico HTTP (FastAPI + Uvicorn).
# Build reproduzivel: as dependencias vem do uv.lock (--frozen).
#
# Observacao: os artefatos do modelo (models/model.pkl e models/model_metadata.json)
# sao versionados pelo DVC. Rode `dvc pull` antes do build para te-los em disco.
# =============================================================================
FROM python:3.12-slim

# uv oficial (mesma versao usada no ambiente local e na CI).
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PROJECT_ROOT=/app

# 1) Dependencias (camada cacheavel): apenas os manifestos do ambiente.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Codigo e artefatos do modelo.
COPY src ./src
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
