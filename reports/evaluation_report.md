# Relatorio de Avaliacao — Deteccao de Fraude (Credicard)

> Gerado automaticamente por `src/config/train_model.py`. Nao editar manualmente.

## Estrategia de validacao
- StratifiedKFold com 10 dobras (shuffle=True, seed=42)
- Calibracao: sigmoid (CalibratedClassifierCV, cv=3)
- Regra de selecao do threshold: `maximize_precision_subject_to_recall`
- Threshold operacional selecionado: 0.566
- Dataset: 284807 linhas | 30 features | 492 fraudes (0.1727%)

## Metricas por dobra (threshold operacional)
| dobra | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9994 | 0.8837 | 0.7755 | 0.8261 | 0.9750 | 0.8180 |
| 2 | 0.9996 | 1.0000 | 0.7755 | 0.8736 | 0.9957 | 0.8768 |
| 3 | 0.9996 | 1.0000 | 0.7959 | 0.8864 | 0.9809 | 0.9150 |
| 4 | 0.9997 | 1.0000 | 0.8163 | 0.8989 | 0.9691 | 0.8907 |
| 5 | 0.9996 | 0.9318 | 0.8367 | 0.8817 | 0.9884 | 0.8525 |
| 6 | 0.9998 | 1.0000 | 0.8600 | 0.9247 | 0.9984 | 0.9196 |
| 7 | 0.9996 | 1.0000 | 0.7800 | 0.8764 | 0.9733 | 0.8573 |
| 8 | 0.9995 | 0.9487 | 0.7551 | 0.8409 | 0.9708 | 0.8313 |
| 9 | 0.9996 | 0.9318 | 0.8367 | 0.8817 | 0.9793 | 0.8409 |
| 10 | 0.9995 | 0.9286 | 0.7959 | 0.8571 | 0.9768 | 0.7839 |

## Resumo entre dobras (media +/- desvio-padrao)
| metrica | media | desvio-padrao |
| --- | --- | --- |
| accuracy | 0.9996 | 0.0001 |
| precision | 0.9625 | 0.0406 |
| recall | 0.8028 | 0.0318 |
| f1 | 0.8748 | 0.0267 |
| roc_auc | 0.9808 | 0.0097 |
| pr_auc | 0.8586 | 0.0407 |

## Metricas out-of-fold (agregadas)
- Threshold 0,500: precision 0.9587 | recall 0.8028 | f1 0.8739
- Threshold 0.566: precision 0.9611 | recall 0.8028 | f1 0.8749
- Livres de threshold: ROC-AUC 0.9804 | PR-AUC 0.8573

## Trade-off precision x recall (melhor precision por piso de recall)
| piso de recall | threshold | precision | recall |
| --- | --- | --- | --- |
| 0.80 | 0.566 | 0.9611 | 0.8028 |
| 0.82 | 0.186 | 0.9140 | 0.8211 |
| 0.85 | 0.026 | 0.6960 | 0.8516 |
| 0.90 | 0.003 | 0.1547 | 0.9004 |

## Matriz de confusao (threshold operacional)
| Real \ Predito | Legitimo | Fraude |
| --- | --- | --- |
| Legitimo | 284299 | 16 |
| Fraude | 97 | 395 |

## Robustez a seed (repeated stratified CV)
| seed | threshold | precision | recall |
| --- | --- | --- | --- |
| 42 | 0.566 | 0.9625 | 0.8028 |
| 7 | 0.491 | 0.9532 | 0.8071 |
| 2024 | 0.556 | 0.9485 | 0.8012 |
| **media** | | **0.9547** | **0.8037** |
| **desvio-padrao** | | 0.0058 | 0.0025 |

## Evidencias graficas
- Curva Precision-Recall: `reports/evaluation_plots/precision_recall_curve.png`
- Matriz de confusao: `reports/evaluation_plots/confusion_matrix.png`
