# Guia do Avaliador

## Objetivo

Este guia resume o caminho mais curto para validar que o repositório está profissional, reproduzível e pronto para discussão com times técnicos.

## Avaliação em 5 minutos

1. Abra o [README](../README.md) para entender a linha do tempo entre `snapshot`, `v1.0.0` e a trilha de update.
2. Verifique os experimentos públicos no DagsHub:
   - [Experiments](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments)
   - [Models](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/models)
3. Consulte as evidências locais:
   - [metrics.json](../metrics.json)
   - [model_metadata.json](../models/model_metadata.json)
   - [data_quality_report.json](../reports/data_quality_report.json)
   - [mlflow_public_registry_summary.json](../reports/mlflow_public_registry_summary.json)
   - [production_readiness_scorecard.json](../reports/production_readiness_scorecard.json)
4. Confirme os contratos automáticos em [tests](../tests).
5. Execute a inferência oficial:

```bash
uv run python -m src.serving.cli --input dados/credicard_tratado.parquet --output predictions.csv --limit 5 --format csv
```

6. Suba a API HTTP de referência e valide o contrato:

```bash
uv run uvicorn src.serving.api:app --port 8000   # GET /health, GET /model/metadata, POST /predict
```

## O que observar

- O threshold operacional está versionado em `models/model_metadata.json`.
- O pipeline exige aceite mínimo de precisão e recall antes de promover o modelo.
- O DVC do branch atual não depende mais de ponteiros órfãos.
- O MLflow/DagsHub público expõe o histórico experimental e o best-model.
- A suíte de testes cobre dados, métricas, inferência, API HTTP de serving, links públicos e relatórios.
- O `ruff` roda sem avisos e é gate obrigatório da CI, junto com os testes, o build da imagem Docker (sem publicar) e o gate de idempotência do DVC.
- A camada de serving (CLI + API FastAPI + imagem Docker) é de **referência**: prova o empacotamento, **sem** deploy, SLA ou monitoramento.

## Leitura recomendada

- [README](../README.md)
- [environment-setup.md](./environment-setup.md)
- [release-update-draft.md](./release-update-draft.md)
