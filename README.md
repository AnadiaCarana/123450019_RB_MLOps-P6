# Praktikum MLOps: Model Experimentation & Tracking (slide 88)

Alur end-to-end: **Fashion-MNIST subset → baseline CNN → Optuna (20 trial, nested runs) → 3 seed → ablasi → registry `@champion` → DVC hash di setiap run.**

## Status kode

Kode ini **belum pernah dijalankan end-to-end** (hanya lolos `py_compile`). Pembuatnya tidak memiliki PyTorch, MLflow, Optuna, ataupun akses jaringan. Jalankan dari awal di mesin Anda, dan bila ada galat, periksa dahulu versi paket (API MLflow 3.x memakai `name=` pada `log_model`; MLflow 2.x memakai `artifact_path=`).

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git init && dvc init
git add . && git commit -m "init"
```

## 2. Data + DVC (langkah 8)

```bash
python src/prepare_data.py          # unduh Fashion-MNIST, buat data/subset/*.npy
dvc add data/subset                 # membuat data/subset.dvc (berisi md5 konten)
git add data/subset.dvc data/.gitignore && git commit -m "track dataset with DVC"
dvc status                          # harus: Data and pipelines are up to date.
```

Hash `md5` dari `data/subset.dvc` dibaca otomatis oleh `common.start_run()` dan dicatat sebagai **tag + parameter `data_md5`** di setiap run (parent dan child). Bila `data/subset.dvc` tidak ada, skrip berhenti dengan pesan galat, bukan mencatat hash kosong.

## 3. MLflow server lokal (langkah 1)

Terminal terpisah, biarkan berjalan:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts \
              --host 127.0.0.1 --port 5000
```

UI: http://127.0.0.1:5000. Skrip memakai `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5000`).

## 4. Urutan eksekusi

```bash
cd src
python train_baseline.py      # langkah 3: baseline, 3 seed, log params/metrik per epoch/model
python tune_optuna.py         # langkah 4: 20 trial, nested runs -> configs/best_config.json
python final_seeds.py         # langkah 5: config terbaik x 3 seed -> mean ± std (+ test set, sekali)
python ablation.py            # langkah 6: full / no_augmentation / no_dropout / no_batchnorm
python register_champion.py   # langkah 7a: registrasi + alias @champion (dipilih via val_acc)
python predict_champion.py    # langkah 7b: muat models:/fmnist-cnn-classifier@champion lalu prediksi
python summarize.py           # tabel hasil dari MLflow -> ../report/results.md
cd ..
git add configs/best_config.json report && git commit -m "results"
```

Estimasi waktu (perkiraan, bergantung CPU/GPU): tuning 20 trial × 10 epoch adalah bagian terlama.

## 5. Screenshot compare view (deliverable)

1. Buka experiment `fmnist-cnn`.
2. Pilih (centang) run `final-best-config-seed11/22/33` dan `ablation-*-seed*`.
3. Klik **Compare**, tangkap layar tampilan perbandingan (parameter + metrik).
4. Tangkap juga tab **Models** yang memperlihatkan alias `champion`.

## 6. Aturan yang dijaga kode

| Aturan | Penerapan |
|---|---|
| Test set tidak dipakai untuk seleksi | Tuning, pemilihan model, dan ablasi memakai validation. Hanya `final_seeds.py` mengisi `test_*` |
| Pemilihan `@champion` | `max val_acc` di antara run `final-seed`, bukan `test_acc` |
| Reproduksibilitas | Seed tetap, `data_md5` di tiap run, konfigurasi disimpan sebagai `config.json` |
| Std | `ddof=1` (simpangan baku sampel) |

## 7. Struktur

```
src/common.py            config, data, model, training, helper MLflow
src/prepare_data.py      subset stratified -> data/subset/*.npy
src/train_baseline.py    src/tune_optuna.py    src/final_seeds.py
src/ablation.py          src/register_champion.py    src/predict_champion.py
src/summarize.py         -> report/results.md
configs/best_config.json (dibuat oleh tune_optuna.py)
report/laporan.md        kerangka laporan + daftar pustaka
```
