# Eksperimen Terlacak dan Terversi untuk Klasifikasi Citra Fashion-MNIST dengan MLflow, Optuna, dan DVC

> **Status dokumen:** kerangka laporan. Semua sel bertanda `⟦ISI⟧` harus diisi dari hasil run Anda sendiri
> (`python src/summarize.py` menghasilkan `report/results.md`). Jangan mengisi angka dari sumber lain.

## 1. Pendahuluan

Jaringan saraf konvolusional (CNN) telah memberi lompatan performa pada pemrosesan citra (LeCun et al., 2015). Namun, performa model di notebook belum cukup: praktik MLOps menekankan versioning, otomasi, dan reproduksibilitas sepanjang siklus hidup model (Kreuzberger et al., 2023). Tanpa pelacakan eksperimen, sistem ML mudah menumpuk utang teknis berupa ketergantungan data dan masalah konfigurasi yang tidak terdokumentasi (Sculley et al., 2015), dan kesulitan muncul pada setiap tahap alur deployment (Paleyes et al., 2022). Reproduksibilitas sendiri didefinisikan sebagai memperoleh hasil serupa dengan kode dan data yang sama (Pineau et al., 2021).

**Tujuan.** Membangun alur eksperimen *end-to-end* yang dapat diaudit: (1) baseline CNN, (2) *hyperparameter tuning* ≥ 20 trial, (3) konfigurasi terbaik dengan 3 *seed* (mean ± std), (4) ablasi ≥ 2 komponen, (5) registrasi model dengan alias `@champion`, dan (6) versioning dataset dengan DVC yang hash-nya tercatat pada setiap run.

## 2. Metode

**Data.** Fashion-MNIST berisi citra abu-abu 28×28 dari 10 kategori produk, dengan 60.000 data latih dan 10.000 data uji (Xiao et al., 2017). Dari data tersebut dibuat subset stratified (seed 42): 10.000 *train*, 2.000 *validation* (dari data latih asli), dan 2.000 *test* (dari data uji asli). Subset disimpan sebagai berkas `.npy` di `data/subset` dan dilacak DVC; hash konten (`md5` dari `data/subset.dvc`) dicatat sebagai *tag* dan *parameter* `data_md5` pada **setiap** run MLflow (parent maupun child).

**Model.** CNN 2 blok konvolusi (3×3, ReLU, max-pool) + 1 lapis *fully connected* 128 unit. Normalisasi input berada di dalam model agar model terdaftar bersifat *self-contained*. Komponen opsional: *batch normalization* (Ioffe & Szegedy, 2015), *dropout* sebelum lapis keluaran (Srivastava et al., 2014), dan augmentasi geometris (*random crop* padding 2 piksel dan *horizontal flip*; Shorten & Khoshgoftaar, 2019). Optimizer Adam, 10 epoch, seluruh metrik per epoch (`train_loss`, `val_loss`, `val_acc`, `val_macro_f1`) dicatat ke MLflow.

**Protokol evaluasi.** Seleksi (tuning, pemilihan seed/model, ablasi) hanya memakai *validation set*. *Test set* dievaluasi **sekali**, hanya pada konfigurasi terbaik (`src/final_seeds.py`). Alasannya: *over-fitting* pada kriteria seleksi dapat berukuran sebanding dengan selisih performa antaralgoritma (Cawley & Talbot, 2010), dan kebocoran data menghasilkan klaim yang terlalu optimistis (Kapoor & Narayanan, 2023).

**Tuning.** Optuna dengan *sampler* TPE (seed 42), 20 trial; satu trial = satu *nested run* MLflow (Akiba et al., 2019). Ruang pencarian: `lr` ∈ [1e-4, 1e-2] (log), `batch_size` ∈ {64, 128, 256}, `dropout` ∈ [0,1; 0,6], `base_channels` ∈ {16, 32, 64}, `weight_decay` ∈ [1e-6, 1e-3] (log); BN dan augmentasi dibuat aktif tetap. Pencarian acak terbukti lebih efisien daripada *grid* dan menjadi baseline alami untuk menilai pencari adaptif (Bergstra & Bengio, 2012); karena itu, bila diperlukan, 20 trial TPE dapat dibandingkan dengan 20 trial acak (`optuna.samplers.RandomSampler`) sebagai pembanding.

**Seed dan ablasi.** Konfigurasi terbaik dijalankan dengan seed {11, 22, 33}; ringkasan memakai rerata dan simpangan baku sampel (ddof = 1). Ablasi *leave-one-out* menonaktifkan satu komponen per varian: tanpa augmentasi, tanpa dropout, tanpa batch normalization, masing-masing 3 seed, dibandingkan pada validation set.

**Registry.** Dari run `final-seed`, model dengan `val_acc` tertinggi (bukan `test_acc`) diregistrasi sebagai `fmnist-cnn-classifier`, diberi alias `@champion`, lalu dimuat dan dipakai memprediksi lewat `models:/fmnist-cnn-classifier@champion`.

## 3. Hipotesis (ditetapkan sebelum melihat hasil)

| ID | Hipotesis | Dasar |
|---|---|---|
| H1 | Konfigurasi terbaik (tuning + BN + dropout + augmentasi) memiliki `val_acc` rerata lebih tinggi daripada baseline | Regularisasi dan penyetelan hyperparameter mengurangi *overfitting* (Srivastava et al., 2014; Shorten & Khoshgoftaar, 2019) |
| H2 | Tanpa augmentasi, `val_acc` turun | Augmentasi memperkaya data latih terbatas (Shorten & Khoshgoftaar, 2019) |
| H3 | Tanpa dropout, penurunan `val_acc` kecil karena BN sudah berperan sebagai regularisator | BN dapat menggantikan dropout pada beberapa kasus (Ioffe & Szegedy, 2015) |
| H4 | Tanpa BN pada `lr` hasil tuning, `val_acc` turun | `lr` dipilih saat BN aktif; BN memungkinkan laju belajar lebih tinggi (Ioffe & Szegedy, 2015) |

**Aturan keputusan (ditetapkan di awal).** Suatu efek dinyatakan *teramati* bila selisih rerata melebihi simpangan baku terbesar kedua kelompok **dan** urutan konsisten pada ketiga seed; selain itu dinyatakan *tidak meyakinkan*. Ini aturan heuristik, bukan uji hipotesis: uji non-parametrik seperti Wilcoxon signed-rank direkomendasikan untuk membandingkan dua pengklasifikasi lintas banyak dataset (Demšar, 2006), sedangkan di sini hanya ada satu dataset dan 3 seed. Pada n = 3, p-value eksak dua sisi paling kecil adalah 0,25, sehingga signifikansi formal tidak mungkin dicapai.

## 4. Hasil

Seluruh angka bersumber dari MLflow (`report/results.md`). Dataset: `data_md5 = ⟦ISI dari data/subset.dvc⟧`.

**Tabel 1. Baseline vs konfigurasi terbaik (mean ± std, 3 seed).**

| Konfigurasi | val_acc | val_macro_f1 | test_acc | test_macro_f1 |
|---|---|---|---|---|
| Baseline | ⟦ISI⟧ | ⟦ISI⟧ | tidak dievaluasi | tidak dievaluasi |
| Terbaik (Optuna) | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |

**Tabel 2. Hyperparameter terbaik hasil 20 trial.** `lr = ⟦ISI⟧`, `batch_size = ⟦ISI⟧`, `dropout = ⟦ISI⟧`, `base_channels = ⟦ISI⟧`, `weight_decay = ⟦ISI⟧`; `val_acc` terbaik satu-seed = ⟦ISI⟧.

**Tabel 3. Ablasi (mean ± std, 3 seed, validation).**

| Varian | val_acc | Δ vs full | Teramati? |
|---|---|---|---|
| full | ⟦ISI⟧ | – | – |
| no_augmentation | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ya / tidak meyakinkan⟧ |
| no_dropout | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ya / tidak meyakinkan⟧ |
| no_batchnorm | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ya / tidak meyakinkan⟧ |

**Gambar 1.** Tangkapan layar *compare view* MLflow UI (pilih run `final-best-config-seed*` dan `ablation-*-seed*`). ⟦sisipkan screenshot⟧

**Gambar 2.** Tangkapan layar tab *Models*: `fmnist-cnn-classifier` dengan alias `champion`. ⟦sisipkan screenshot⟧

## 5. Pembahasan

⟦ISI: untuk tiap H1–H4, tulis "didukung / tidak didukung / tidak meyakinkan" menurut aturan keputusan di Bagian 3, beserta angka dari Tabel 1 dan 3.⟧

**Keterbatasan yang wajib dicantumkan.**

1. *Hanya 3 seed.* Galat evaluasi pada sampel kecil cenderung lebih besar daripada yang tampak, dan simpangan baku antar-lipatan atau antar-run meremehkannya (Varoquaux, 2018). Pada validation set 2.000 citra dengan akurasi sekitar 0,90, galat baku binomial sekitar 0,007 (≈ 0,7 poin persen), sehingga selisih akurasi di bawah kira-kira 1 poin persen sulit dibedakan dari derau sampling.
2. *Seleksi dan evaluasi memakai data yang berbeda,* tetapi tuning satu-seed pada validation set tetap dapat menyetel diri terhadap derau validation (Cawley & Talbot, 2010).
3. *Subset kecil dan 10 epoch.* Efek augmentasi dan dropout biasanya baru terlihat jelas pada pelatihan lebih panjang; hasil ablasi tidak boleh digeneralisasi ke dataset atau arsitektur lain.
4. *Determinisme.* Pada GPU, hasil dapat berbeda antarmesin meski seed sama; reproduksi penuh menuntut commit Git, hash data, dan lingkungan yang sama (Pineau et al., 2021).

## 6. Kesimpulan

⟦ISI: 3–4 kalimat. Jawab langsung: (a) apakah H1 didukung dan seberapa besar selisihnya relatif terhadap std, (b) komponen mana pada ablasi yang efeknya teramati, (c) apakah model `@champion` berhasil dimuat dan memprediksi. Jangan menulis klaim yang tidak ditopang Tabel 1–3.⟧

## 7. Reproduksi

```bash
git clone ⟦URL repositori Anda⟧ && cd mlops-praktikum
pip install -r requirements.txt
dvc pull            # atau: python src/prepare_data.py && dvc add data/subset
dvc status          # harus: Data and pipelines are up to date.
```

Urutan eksekusi lengkap ada di `README.md`.

## Daftar Pustaka

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of KDD 2019 (Applied Data Science track)*. https://arxiv.org/pdf/1907.10902

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*(10), 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*(70), 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*(1), 1–30. https://www.jmlr.org/papers/v7/demsar06a.html

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning, 37*, 448–456. https://proceedings.mlr.press/v37/ioffe15.html

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856

Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, and architecture. *IEEE Access, 11*, 31866–31879. https://ieeexplore.ieee.org/document/10081336

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444. https://www.nature.com/articles/nature14539

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys*. https://arxiv.org/abs/2011.09926v3

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems, 28*, 2503–2511. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*, Article 60. https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-019-0197-0

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html

Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage, 180*, 68–77. https://arxiv.org/pdf/1706.07581

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms*. arXiv. https://arxiv.org/abs/1708.07747v1
