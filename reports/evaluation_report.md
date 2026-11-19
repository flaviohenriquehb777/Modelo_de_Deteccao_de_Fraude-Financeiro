# Relatorio de Avaliacao — Deteccao de Fraude (Credicard)

> Gerado automaticamente por `src/config/train_model.py`. Nao editar manualmente.

## Estrategia de validacao
- StratifiedKFold com 10 dobras (shuffle=True, seed=42)
- Threshold operacional selecionado: 0.565
- Dataset: 284807 linhas | 30 features | 492 fraudes (0.1727%)

## Metricas por dobra (threshold operacional)
| dobra | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9993 | 0.8125 | 0.7959 | 0.8041 | 0.9750 | 0.8180 |
| 2 | 0.9996 | 0.9744 | 0.7755 | 0.8636 | 0.9957 | 0.8768 |
| 3 | 0.9997 | 0.9762 | 0.8367 | 0.9011 | 0.9809 | 0.9150 |
| 4 | 0.9997 | 1.0000 | 0.8367 | 0.9111 | 0.9691 | 0.8907 |
| 5 | 0.9996 | 0.9111 | 0.8367 | 0.8723 | 0.9884 | 0.8525 |
| 6 | 0.9998 | 1.0000 | 0.8600 | 0.9247 | 0.9984 | 0.9196 |
| 7 | 0.9994 | 0.8864 | 0.7800 | 0.8298 | 0.9733 | 0.8573 |
| 8 | 0.9994 | 0.9024 | 0.7551 | 0.8222 | 0.9708 | 0.8313 |
| 9 | 0.9996 | 0.9130 | 0.8571 | 0.8842 | 0.9793 | 0.8409 |
| 10 | 0.9995 | 0.8864 | 0.7959 | 0.8387 | 0.9768 | 0.7839 |

## Resumo entre dobras (media +/- desvio-padrao)
| metrica | media | desvio-padrao |
| --- | --- | --- |
| accuracy | 0.9996 | 0.0001 |
| precision | 0.9262 | 0.0573 |
| recall | 0.8130 | 0.0350 |
| f1 | 0.8652 | 0.0386 |
| roc_auc | 0.9808 | 0.0097 |
| pr_auc | 0.8586 | 0.0407 |

## Metricas out-of-fold (agregadas)
- Threshold 0,500: precision 0.9178 | recall 0.8171 | f1 0.8645
- Threshold 0.565: precision 0.9238 | recall 0.8130 | f1 0.8649
- Livres de threshold: ROC-AUC 0.9804 | PR-AUC 0.8573

## Matriz de confusao (threshold operacional)
| Real \ Predito | Legitimo | Fraude |
| --- | --- | --- |
| Legitimo | 284282 | 33 |
| Fraude | 92 | 400 |

## Robustez a seed (repeated stratified CV)
| seed | threshold | precision | recall |
| --- | --- | --- | --- |
| 42 | 0.565 | 0.9262 | 0.8130 |
| 7 | 0.545 | 0.9241 | 0.8173 |
| 2024 | 0.515 | 0.9236 | 0.8215 |
| **media** | | **0.9246** | **0.8172** |
| **desvio-padrao** | | 0.0011 | 0.0035 |

## Evidencias graficas
- Curva Precision-Recall: `reports/evaluation_plots/precision_recall_curve.png`
- Matriz de confusao: `reports/evaluation_plots/confusion_matrix.png`
