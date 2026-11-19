# Benchmark multi-modelo — Deteccao de Fraude (Credicard)

> Gerado automaticamente por `src/config/benchmark_models.py`. Nao editar manualmente.

## Protocolo honesto (anti-leakage)
- StratifiedKFold com 10 dobras (seed=42)
- Regra de threshold: `maximize_precision_subject_to_recall` com piso de recall 0.80
- Calibracao: sigmoid (global, unica, sobre o out-of-fold)
- Dataset: 284807 linhas | 30 features | 492 fraudes (0.1727%)

## Ranking (maior precision factivel no piso de recall)
| # | familia | PR-AUC | ROC-AUC | precision (OOF) | recall (OOF) | thr |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | lightgbm | 0.8596 | 0.9785 | 0.9638 | 0.8110 | 0.648 |
| 2 | xgboost | 0.8573 | 0.9804 | 0.9611 | 0.8028 | 0.566 |
| 3 | random_forest | 0.8351 | 0.9801 | 0.8800 | 0.8049 | 0.211 |
| 4 | logistic_regression | 0.7376 | 0.9789 | 0.8243 | 0.8008 | 0.060 |

Vencedor do benchmark: **lightgbm**.

## Trade-off precision x recall por familia
| familia | piso 0.80 | piso 0.82 | piso 0.85 | piso 0.90 |
| --- | --- | --- | --- | --- |
| lightgbm | 0.9638 | 0.9352 | 0.8216 | 0.0992 |
| xgboost | 0.9611 | 0.9140 | 0.6960 | 0.1547 |
| random_forest | 0.8800 | 0.8614 | 0.7403 | 0.0781 |
| logistic_regression | 0.8243 | 0.6495 | 0.3883 | 0.0820 |

## Tempo de execucao por familia (s)

- xgboost: 43.3s
- lightgbm: 35.2s
- random_forest: 346.6s
- logistic_regression: 18.3s

## Busca restrita (validacao aninhada)

- Familia tunada: **lightgbm**
- Espaco de busca: 648 combinacoes | n_iter=10 | outer=5 | inner=3
- Metrica honesta (OOF aninhado): PR-AUC 0.8449 | ROC-AUC 0.9555 | precision 0.9568 | recall 0.8110 @ thr 0.280

### Hiperparametros selecionados por dobra externa

| dobra | parametros |
| --- | --- |
| 1 | classifier__colsample_bytree=0.6, classifier__learning_rate=0.2, classifier__min_child_samples=50, classifier__n_estimators=200, classifier__num_leaves=31, classifier__scale_pos_weight=577.8760162601626 |
| 2 | classifier__colsample_bytree=0.6, classifier__learning_rate=0.1, classifier__min_child_samples=50, classifier__n_estimators=200, classifier__num_leaves=127, classifier__scale_pos_weight=577.8760162601626 |
| 3 | classifier__colsample_bytree=0.6, classifier__learning_rate=0.05, classifier__min_child_samples=50, classifier__n_estimators=200, classifier__num_leaves=31, classifier__scale_pos_weight=1.0 |
| 4 | classifier__colsample_bytree=0.6, classifier__learning_rate=0.2, classifier__min_child_samples=50, classifier__n_estimators=200, classifier__num_leaves=31, classifier__scale_pos_weight=577.8760162601626 |
| 5 | classifier__colsample_bytree=0.6, classifier__learning_rate=0.05, classifier__min_child_samples=50, classifier__n_estimators=200, classifier__num_leaves=31, classifier__scale_pos_weight=1.0 |
