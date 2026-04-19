# Requirements: Sistem Klasifikasi Kematangan Tomat Berbasis Deep Learning

## Introduction

Dokumen ini mendefinisikan kebutuhan fungsional dan non-fungsional untuk sistem klasifikasi kematangan tomat berbasis Transfer Learning MobileNetV2. Sistem ini dikembangkan sebagai bagian dari penelitian jurnal akademik (target Sinta 2/3) untuk mendukung otomasi grading kematangan tomat pada UMKM pertanian.

---

## Requirements

### 1. Data Pipeline

#### 1.1 Pembacaan Anotasi CSV

**User Story**: Sebagai peneliti, saya ingin sistem dapat membaca file `_annotations.csv` dari dataset Roboflow sehingga informasi bounding box tersedia untuk proses crop ROI.

**Acceptance Criteria**:
- [ ] 1.1.1 Fungsi `load_annotations(csv_path)` membaca file CSV dengan kolom: `filename, width, height, class, xmin, ymin, xmax, ymax`
- [ ] 1.1.2 Fungsi mengembalikan `pd.DataFrame` dengan tipe data yang benar (int untuk koordinat, str untuk filename)
- [ ] 1.1.3 Fungsi menangani file CSV yang tidak ditemukan dengan `FileNotFoundError` yang deskriptif
- [ ] 1.1.4 Fungsi mendukung ketiga split dataset: `dataset/train/`, `dataset/valid/`, `dataset/test/`

#### 1.2 Crop ROI dari Bounding Box

**User Story**: Sebagai peneliti, saya ingin sistem dapat mengekstrak Region of Interest (ROI) tomat dari gambar berdasarkan koordinat bounding box sehingga input klasifikasi hanya berisi area tomat.

**Acceptance Criteria**:
- [ ] 1.2.1 Fungsi `crop_and_save_rois()` menghasilkan gambar crop berukuran tepat 224x224 pixel
- [ ] 1.2.2 Gambar crop disimpan dalam struktur direktori `output_dir/{class_label}/` sesuai mapping kelas
- [ ] 1.2.3 Mapping kelas: 0 → "Matang", 1 → "Setengah Matang", 2 → "Mentah"
- [ ] 1.2.4 Gambar dengan ROI < 10x10 pixel di-skip dengan log warning (tidak menyebabkan crash)
- [ ] 1.2.5 Gambar sumber tidak dimodifikasi (read-only)
- [ ] 1.2.6 Fungsi memproses semua 1.006 gambar dari ketiga split tanpa error fatal

#### 1.3 Data Generator dengan Augmentasi

**User Story**: Sebagai peneliti, saya ingin data generator dengan augmentasi yang tepat sehingga model lebih robust terhadap variasi kondisi lapangan.

**Acceptance Criteria**:
- [ ] 1.3.1 `create_data_generators()` menghasilkan generator dengan augmentasi: rotation_range=20, zoom_range=0.2, horizontal_flip=True
- [ ] 1.3.2 Normalisasi pixel ke range [0, 1] via `rescale=1./255`
- [ ] 1.3.3 Validation generator tidak menggunakan augmentasi (hanya rescale)
- [ ] 1.3.4 `create_test_generator()` menghasilkan generator dengan `shuffle=False` untuk evaluasi yang reproducible
- [ ] 1.3.5 Batch shape output: `(batch_size, 224, 224, 3)`
- [ ] 1.3.6 Augmentasi tidak mengubah distribusi label (jumlah sampel per kelas tetap proporsional)

---

### 2. Arsitektur Model

#### 2.1 Transfer Learning MobileNetV2

**User Story**: Sebagai peneliti, saya ingin menggunakan MobileNetV2 pretrained ImageNet sebagai base model sehingga sistem dapat mencapai akurasi tinggi dengan dataset yang relatif kecil (1.006 gambar).

**Acceptance Criteria**:
- [ ] 2.1.1 `build_mobilenetv2_model()` menggunakan MobileNetV2 dengan `weights="imagenet"` dan `include_top=False`
- [ ] 2.1.2 Semua layer base MobileNetV2 di-freeze (`trainable=False`) pada tahap awal training
- [ ] 2.1.3 Custom head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(3, softmax)`
- [ ] 2.1.4 Model dikompilasi dengan `Adam(lr=0.001)`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`
- [ ] 2.1.5 Output layer memiliki 3 neuron sesuai jumlah kelas

#### 2.2 Fine-Tuning (Opsional)

**User Story**: Sebagai peneliti, saya ingin opsi fine-tuning layer terakhir MobileNetV2 sehingga model dapat beradaptasi lebih baik dengan karakteristik visual tomat.

**Acceptance Criteria**:
- [ ] 2.2.1 `unfreeze_top_layers(model, num_layers=20)` membuat N layer terakhir base model trainable
- [ ] 2.2.2 Learning rate dikurangi menjadi 1e-5 untuk fine-tuning (mencegah catastrophic forgetting)
- [ ] 2.2.3 Fine-tuning hanya dilakukan setelah training tahap 1 selesai

---

### 3. Training

#### 3.1 Training Loop dengan Callbacks

**User Story**: Sebagai peneliti, saya ingin training otomatis berhenti saat model tidak lagi membaik sehingga mencegah overfitting dan menghemat waktu komputasi.

**Acceptance Criteria**:
- [ ] 3.1.1 `get_callbacks()` mengembalikan list yang berisi minimal: `EarlyStopping` dan `ModelCheckpoint`
- [ ] 3.1.2 `EarlyStopping`: `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`
- [ ] 3.1.3 `ModelCheckpoint`: simpan model terbaik berdasarkan `val_accuracy` ke `output/models/best_model.h5`
- [ ] 3.1.4 `ReduceLROnPlateau`: `monitor='val_loss'`, `factor=0.5`, `patience=5`
- [ ] 3.1.5 `train_model()` menjalankan training maksimal 50 epoch
- [ ] 3.1.6 Fungsi mengembalikan `History` object untuk keperluan plotting

#### 3.2 Reproducibility

**User Story**: Sebagai peneliti, saya ingin hasil training yang reproducible sehingga eksperimen dapat diulang dengan hasil yang konsisten untuk keperluan jurnal.

**Acceptance Criteria**:
- [ ] 3.2.1 Random seed di-set di awal (`tf.random.set_seed(42)`, `np.random.seed(42)`)
- [ ] 3.2.2 Konfigurasi training (epochs, batch_size, learning_rate) dapat dikonfigurasi via parameter

---

### 4. Evaluasi

#### 4.1 Visualisasi Training History

**User Story**: Sebagai peneliti, saya ingin grafik akurasi dan loss training vs validasi sehingga dapat menganalisis proses pembelajaran model untuk laporan jurnal.

**Acceptance Criteria**:
- [ ] 4.1.1 `plot_training_history()` menghasilkan figure dengan 2 subplot: akurasi dan loss
- [ ] 4.1.2 Setiap subplot menampilkan kurva training (biru) dan validasi (oranye)
- [ ] 4.1.3 Grafik disimpan sebagai PNG ke `output/plots/training_history.png`
- [ ] 4.1.4 Grafik memiliki judul, label sumbu, dan legenda dalam Bahasa Indonesia

#### 4.2 Confusion Matrix

**User Story**: Sebagai peneliti, saya ingin confusion matrix yang divisualisasikan sehingga dapat menganalisis pola kesalahan klasifikasi per kelas.

**Acceptance Criteria**:
- [ ] 4.2.1 `generate_confusion_matrix()` menghasilkan heatmap dengan seaborn
- [ ] 4.2.2 Label sumbu menggunakan nama kelas: ["Matang", "Setengah Matang", "Mentah"]
- [ ] 4.2.3 Confusion matrix disimpan ke `output/plots/confusion_matrix.png`
- [ ] 4.2.4 Nilai dalam confusion matrix menampilkan jumlah absolut dan persentase

#### 4.3 Classification Report

**User Story**: Sebagai peneliti, saya ingin classification report dengan Precision, Recall, dan F1-Score per kelas sehingga dapat melaporkan performa model secara komprehensif di jurnal.

**Acceptance Criteria**:
- [ ] 4.3.1 `generate_classification_report()` menghasilkan report dengan Precision, Recall, F1-Score untuk setiap kelas
- [ ] 4.3.2 Report mencakup macro average dan weighted average
- [ ] 4.3.3 Report disimpan sebagai file teks ke `output/reports/classification_report.txt`
- [ ] 4.3.4 Fungsi mengembalikan dict untuk akses programatik

#### 4.4 Threshold Akurasi Penelitian

**User Story**: Sebagai peneliti, saya ingin sistem mencapai akurasi minimal 80% pada test set sehingga hasil penelitian layak dipublikasikan di jurnal Sinta 2/3.

**Acceptance Criteria**:
- [ ] 4.4.1 Akurasi keseluruhan pada test set ≥ 80%
- [ ] 4.4.2 F1-Score untuk setiap kelas ≥ 0.75
- [ ] 4.4.3 Hasil evaluasi mencantumkan akurasi test set secara eksplisit

---

### 5. Inferensi

#### 5.1 Fungsi grading_tomat

**User Story**: Sebagai petani/operator UMKM, saya ingin fungsi sederhana yang menerima path gambar tomat dan mengembalikan label kematangan sehingga dapat diintegrasikan ke aplikasi mobile atau sistem sortasi.

**Acceptance Criteria**:
- [ ] 5.1.1 `grading_tomat(image_path)` menerima path gambar JPG atau PNG dengan ukuran bebas
- [ ] 5.1.2 Fungsi mengembalikan dict dengan key: `label`, `confidence`, `probabilities`
- [ ] 5.1.3 `label` selalu berupa salah satu dari: "Matang", "Setengah Matang", "Mentah"
- [ ] 5.1.4 `confidence` selalu dalam range [0.0, 1.0]
- [ ] 5.1.5 Jumlah semua nilai dalam `probabilities` = 1.0 (±1e-5)
- [ ] 5.1.6 Fungsi tidak crash untuk gambar valid berformat JPG/PNG dengan ukuran apapun (minimal 10x10 pixel)
- [ ] 5.1.7 Preprocessing gambar yang sama selalu menghasilkan tensor yang identik (deterministic)

#### 5.2 Load Model

**User Story**: Sebagai developer, saya ingin fungsi load model yang robust sehingga model dapat dimuat dari berbagai format penyimpanan.

**Acceptance Criteria**:
- [ ] 5.2.1 `load_model()` mendukung format `.h5` dan SavedModel directory
- [ ] 5.2.2 Fungsi mengembalikan `FileNotFoundError` yang deskriptif jika model tidak ditemukan
- [ ] 5.2.3 Model di-cache setelah load pertama untuk efisiensi inferensi berulang

---

### 6. Dokumentasi dan Kode

#### 6.1 Komentar Bahasa Indonesia

**User Story**: Sebagai mahasiswa yang akan sidang, saya ingin semua komentar kode dalam Bahasa Indonesia sehingga dosen pembimbing dapat memahami implementasi dengan mudah.

**Acceptance Criteria**:
- [ ] 6.1.1 Semua docstring fungsi ditulis dalam Bahasa Indonesia
- [ ] 6.1.2 Komentar inline pada bagian kritis ditulis dalam Bahasa Indonesia
- [ ] 6.1.3 Nama variabel menggunakan Bahasa Inggris (konvensi Python) tetapi komentar dalam Bahasa Indonesia

#### 6.2 Entry Point dan Requirements

**User Story**: Sebagai pengguna baru, saya ingin satu file `main.py` yang menjalankan seluruh pipeline sehingga dapat mereproduksi hasil penelitian dengan satu perintah.

**Acceptance Criteria**:
- [ ] 6.2.1 `main.py` menjalankan pipeline lengkap: preprocessing → training → evaluasi
- [ ] 6.2.2 `requirements.txt` mencantumkan semua dependensi dengan versi minimum
- [ ] 6.2.3 Sistem dapat berjalan di Google Colab (Python 3.8+, TensorFlow 2.10+)
- [ ] 6.2.4 `main.py` menyediakan argumen CLI untuk skip tahap tertentu (misal: `--skip-preprocessing` jika ROI sudah di-crop)

#### 6.3 Struktur Output

**User Story**: Sebagai peneliti, saya ingin output terorganisir dalam folder yang jelas sehingga mudah menemukan model, grafik, dan laporan.

**Acceptance Criteria**:
- [ ] 6.3.1 Model tersimpan di `output/models/best_model.h5`
- [ ] 6.3.2 Grafik tersimpan di `output/plots/` (training_history.png, confusion_matrix.png)
- [ ] 6.3.3 Laporan tersimpan di `output/reports/classification_report.txt`
- [ ] 6.3.4 Direktori output dibuat otomatis jika belum ada
