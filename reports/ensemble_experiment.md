# Experimento de ensemble/stacking — risco da meta 96/82 (Fase 8)

> Gerado automaticamente por `src/config/ensemble_experiment.py`. Nao editar manualmente.

## Protocolo honesto (anti-leakage)
- StratifiedKFold com 10 dobras (seed=42)
- Regra de threshold: `maximize_precision_subject_to_recall`
- Calibracao: sigmoid (global, unica, sobre o out-of-fold)
- Dataset: 284807 linhas | 30 features | 492 fraudes (0.1727%)
- Meta aspiracional (NAO bloqueante): precision 0.96 com recall 0.82

## Estrategias avaliadas (out-of-fold honesto)
| estrategia | tipo | PR-AUC | ROC-AUC | precision (OOF) | recall (OOF) | P @ R=0.82 | gap vs 0.96 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | single | 0.8573 | 0.9804 | 0.9611 | 0.8028 | 0.9140 | 0.0460 |
| lightgbm | single | 0.8596 | 0.9785 | 0.9638 | 0.8110 | 0.9352 | 0.0248 |
| random_forest | single | 0.8351 | 0.9801 | 0.8800 | 0.8049 | 0.8614 | 0.0986 |
| logistic_regression | single | 0.7376 | 0.9789 | 0.8243 | 0.8008 | 0.6495 | 0.3105 |
| blend_mean | ensemble | 0.8566 | 0.9828 | 0.9448 | 0.8008 | 0.8728 | 0.0872 |
| stacking_logreg | stacking | 0.8625 | 0.9814 | 0.9610 | 0.8008 | 0.9287 | 0.0313 |
| stacking_logreg_balanced | stacking | 0.8580 | 0.9819 | 0.9144 | 0.8028 | 0.8728 | 0.0872 |

## Veredito
- Melhor modelo unico: **lightgbm** (PR-AUC 0.8596)
- Melhor ensemble: **stacking_logreg** (PR-AUC 0.8625)
- Melhor estrategia geral: **stacking_logreg**
- Ensemble move a fronteira? **nao**
- Meta 96/82 alcancavel? **nao**
- Melhor ponto factivel (recall >= 0.82): precision 0.9352 | gap vs 0.96 = 0.0248

> Nenhuma estrategia (modelo unico, blend ou stacking) atinge 96/82 simultaneamente na fronteira honesta. A meta aspiracional permanece NAO bloqueante: mantem-se o alvo de negocio 96/80, com o gap documentado explicitamente.

## Notas
- Features adicionais foram sondadas na Fase 2 (temporais ciclicas: ganho dentro de 1 desvio-padrao) e descartadas como lever.
- A meta 96/82 simultanea foi confirmada inalcancavel tambem por ensemble/stacking; nao ha vazamento de dados em nenhuma metrica.
