# Tasks: Sistem Klasifikasi Kematangan Tomat Berbasis Deep Learning

## Task List

- [x] 1 Setup Proyek dan Dependensi
  - [x] 1.1 Buat file `requirements.txt` dengan semua dependensi (tensorflow, numpy, pandas, Pillow, matplotlib, seaborn, scikit-learn, hypothesis)
  - [x] 1.2 Buat struktur direktori proyek: `src/`, `output/models/`, `output/plots/`, `output/reports/`
  - [x] 1.3 Buat file `src/utils.py` dengan konstanta global (CLASS_MAPPING, CLASS_NAMES, INPUT_SHAPE, dll)

- [x] 2 Implementasi Data Pipeline (`src/data_pipeline.py`)
  - [x] 2.1 Implementasi fungsi `load_annotations(csv_path)` — baca CSV Roboflow ke DataFrame
  - [x] 2.2 Implementasi fungsi `crop_and_save_rois(annotations_df, img_dir, output_dir, target_size)` — crop ROI dari bounding box dan simpan per kelas
  - [x] 2.3 Implementasi fungsi `create_data_generators(train_dir, valid_dir, batch_size)` — ImageDataGenerator dengan augmentasi untuk training
  - [x] 2.4 Implementasi fungsi `create_test_generator(test_dir, batch_size)` — generator tanpa augmentasi untuk evaluasi

- [x] 3 Implementasi Arsitektur Model (`src/model.py`)
  - [x] 3.1 Implementasi fungsi `build_mobilenetv2_model(num_classes, input_shape, learning_rate)` — MobileNetV2 pretrained + custom head
  - [x] 3.2 Implementasi fungsi `unfreeze_top_layers(model, num_layers)` — untuk fine-tuning opsional tahap 2

- [x] 4 Implementasi Training (`src/train.py`)
  - [x] 4.1 Implementasi fungsi `get_callbacks(model_save_path)` — EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
  - [x] 4.2 Implementasi fungsi `train_model(model, train_gen, valid_gen, epochs)` — training loop utama
  - [x] 4.3 Implementasi fungsi `fine_tune_model(model, train_gen, valid_gen, epochs)` — fine-tuning opsional

- [x] 5 Implementasi Evaluasi (`src/evaluate.py`)
  - [x] 5.1 Implementasi fungsi `plot_training_history(history, save_path)` — grafik akurasi/loss dengan label Bahasa Indonesia
  - [x] 5.2 Implementasi fungsi `generate_confusion_matrix(model, test_gen, class_names, save_path)` — heatmap confusion matrix
  - [x] 5.3 Implementasi fungsi `generate_classification_report(model, test_gen, class_names, save_path)` — Precision/Recall/F1 per kelas
  - [x] 5.4 Implementasi fungsi `evaluate_model(model, test_gen, class_names, output_dir)` — wrapper evaluasi lengkap

- [x] 6 Implementasi Inferensi (`src/inference.py`)
  - [x] 6.1 Implementasi fungsi `load_model(model_path)` — load model .h5 atau SavedModel dengan caching
  - [x] 6.2 Implementasi fungsi `preprocess_image(image_path, target_size)` — preprocessing gambar untuk inferensi
  - [x] 6.3 Implementasi fungsi `grading_tomat(image_path, model_path)` — fungsi utama prediksi kematangan tomat

- [x] 7 Implementasi Entry Point (`main.py`)
  - [x] 7.1 Implementasi `main.py` dengan argparse untuk menjalankan full pipeline (preprocessing → training → evaluasi)
  - [x] 7.2 Tambahkan flag `--skip-preprocessing` untuk skip crop ROI jika sudah dilakukan sebelumnya
  - [x] 7.3 Tambahkan flag `--fine-tune` untuk mengaktifkan fine-tuning tahap 2 setelah training awal

- [ ] 8 Property-Based Testing (`tests/test_properties.py`)
  - [ ] 8.1 [PBT] Tulis property test: output `grading_tomat()` selalu mengembalikan label valid dari {"Matang", "Setengah Matang", "Mentah"}
  - [ ] 8.2 [PBT] Tulis property test: nilai `confidence` selalu dalam range [0.0, 1.0]
  - [ ] 8.3 [PBT] Tulis property test: jumlah semua `probabilities` = 1.0 (±1e-5)
  - [ ] 8.4 [PBT] Tulis property test: `preprocess_image()` bersifat deterministik (input sama → output sama)
  - [ ] 8.5 [PBT] Tulis property test: fungsi tidak crash untuk gambar JPG/PNG valid dengan ukuran sembarang (10-4000 pixel)

- [ ] 9 Unit Testing (`tests/test_units.py`)
  - [ ] 9.1 Tulis unit test untuk `load_annotations()` — verifikasi kolom dan tipe data DataFrame
  - [ ] 9.2 Tulis unit test untuk `crop_and_save_rois()` — verifikasi ukuran output 224x224
  - [ ] 9.3 Tulis unit test untuk `build_mobilenetv2_model()` — verifikasi output shape dan jumlah kelas
  - [ ] 9.4 Tulis unit test untuk `preprocess_image()` — verifikasi shape (1, 224, 224, 3) dan range [0, 1]
  - [ ] 9.5 Tulis unit test untuk `grading_tomat()` — verifikasi struktur dict output

- [ ] 10 Verifikasi Threshold Akurasi
  - [ ] 10.1 Jalankan training penuh dan verifikasi akurasi test set ≥ 80%
  - [ ] 10.2 Verifikasi F1-Score setiap kelas ≥ 0.75
  - [ ] 10.3 Simpan hasil evaluasi final ke `output/reports/final_results.txt`
