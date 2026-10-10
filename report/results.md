# Hasil (otomatis dari MLflow)


## Baseline (3 seed, validation)

| Run | val_acc | val_macro_f1 |
|---|---|---|
| baseline-cnn | 0.8830 ± 0.0026 | 0.8818 ± 0.0043 |

## Konfigurasi terbaik (3 seed)

| Run | val_acc | val_macro_f1 | test_acc | test_macro_f1 |
|---|---|---|---|---|
| final-best-config | 0.8542 ± 0.0035 | 0.8503 ± 0.0039 | 0.8600 ± 0.0048 | 0.8579 ± 0.0046 |

## Ablasi (3 seed, validation)

| Run | val_acc | val_macro_f1 |
|---|---|---|
| ablation-no_batchnorm | 0.8637 ± 0.0123 | 0.8610 ± 0.0147 |
| ablation-no_dropout | 0.8670 ± 0.0064 | 0.8648 ± 0.0088 |
| ablation-no_augmentation | 0.8810 ± 0.0036 | 0.8792 ± 0.0048 |
| ablation-full | 0.8527 ± 0.0118 | 0.8478 ± 0.0119 |

## Tuning

Jumlah trial tercatat: 20
