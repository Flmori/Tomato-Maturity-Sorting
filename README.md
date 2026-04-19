# Sistem Klasifikasi Kematangan Tomat Berbasis Deep Learning

**Riset Jurnal Kewirausahaan - STTR Informatika**

Judul: *"Otomasi Grading Kematangan Tomat Berbasis Deep Learning untuk Efisiensi Operasional UMKM Pertanian"*

---

## 📋 Deskripsi Proyek

Sistem ini mengimplementasikan klasifikasi kematangan tomat secara otomatis menggunakan **Transfer Learning** dengan arsitektur **MobileNetV2** yang telah dilatih pada dataset ImageNet. Sistem dirancang untuk mendukung otomasi grading kematangan tomat pada UMKM pertanian, menggantikan proses sortasi manual yang tidak konsisten dan memakan waktu.

### Kelas Kematangan
- **Matang** (Ripe) - Tomat merah penuh
- **Setengah Matang** (Half-ripe) - Tomat oranye/merah muda
- **Mentah** (Unripe) - Tomat hijau

---

## 🎯 Mengapa MobileNetV2?

Dari perspektif **kewirausahaan UMKM**, MobileNetV2 dipilih karena:

| Kriteria | MobileNetV2 | VGG16/ResNet50 |
|---|---|---|
| Ukuran model | ~14 MB | ~500 MB / ~100 MB |
| Parameter | ~3.4 juta | ~138 juta / ~25 juta |
| Inferensi di CPU | ~50ms/gambar | >500ms/gambar |
| Bisa jalan di HP Android | ✅ Ya (TFLite) | ❌ Tidak praktis |
| Biaya deployment | Gratis (HP petani) | Butuh server cloud |

**Justifikasi bisnis**: Petani UMKM tidak memiliki infrastruktur server. MobileNetV2 memungkinkan deployment langsung di smartphone Android melalui TensorFlow Lite, sehingga sistem grading dapat digunakan di lapangan tanpa koneksi internet.

---

## 📁 Struktur Proyek

```
tomato-ripeness-classifier/
├── dataset/                    # Dataset Roboflow (1.006 gambar)
│   ├── train/                  # 705 gambar + _annotations.csv
│   ├── valid/                  # 201 gambar + _annotations.csv
│   └── test/                   # 100 gambar + _annotations.csv
├── src/                        # Source code
│   ├── utils.py                # Konstanta global dan helper functions
│   ├── data_pipeline.py        # Preprocessing + augmentasi
│   ├── model.py                # Arsitektur MobileNetV2
│   ├── train.py                # Training loop + callbacks
│   ├── evaluate.py             # Evaluasi + visualisasi
│   └── inference.py            # Fungsi grading_tomat()
├── output/                     # Output hasil training
│   ├── models/                 # Saved model (.h5)
│   ├── plots/                  # Grafik akurasi/loss, confusion matrix
│   ├── reports/                # Classification report
│   └── cropped/                # ROI yang sudah di-crop
├── main.py                     # Entry point full pipeline
├── requirements.txt            # Dependensi Python
└── README.md                   # Dokumentasi ini
```

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd tomato-ripeness-classifier
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

**Dependensi utama:**
- TensorFlow >= 2.10.0
- NumPy, Pandas
- Pillow (image processing)
- Matplotlib, Seaborn (visualisasi)
- Scikit-learn (evaluasi)

### 3. Verifikasi Dataset

Pastikan struktur dataset sudah benar:
```
dataset/
  train/
    _annotations.csv
    *.jpg (705 gambar)
  valid/
    _annotations.csv
    *.jpg (201 gambar)
  test/
    _annotations.csv
    *.jpg (100 gambar)
```

---

## 🎓 Cara Menggunakan

### Opsi 1: Jalankan Full Pipeline (Recommended)

```bash
python main.py
```

Pipeline lengkap akan menjalankan:
1. **Preprocessing**: Crop ROI dari bounding box
2. **Training**: Transfer Learning MobileNetV2 (max 50 epoch)
3. **Evaluasi**: Confusion Matrix, Classification Report, Grafik

### Opsi 2: Jalankan dengan Custom Arguments

```bash
# Training dengan 30 epoch dan batch size 16
python main.py --epochs 30 --batch-size 16

# Skip preprocessing jika ROI sudah di-crop
python main.py --skip-preprocessing

# Aktifkan fine-tuning setelah training awal
python main.py --fine-tune

# Skip training dan hanya evaluasi (jika model sudah ada)
python main.py --skip-training --skip-preprocessing
```

**Arguments yang tersedia:**
- `--skip-preprocessing`: Skip crop ROI (jika sudah dilakukan)
- `--skip-training`: Skip training (jika model sudah ada)
- `--skip-evaluation`: Skip evaluasi
- `--fine-tune`: Aktifkan fine-tuning tahap 2
- `--epochs N`: Jumlah epoch (default: 50)
- `--batch-size N`: Batch size (default: 32)
- `--learning-rate F`: Learning rate (default: 0.001)

---

## 🔍 Inferensi (Prediksi Gambar Baru)

Setelah training selesai, gunakan fungsi `grading_tomat()` untuk prediksi:

```python
from src.inference import grading_tomat

# Prediksi kematangan tomat
result = grading_tomat('path/to/tomato_image.jpg')

print(f"Kematangan: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

**Output contoh:**
```python
{
    "label": "Matang",
    "confidence": 0.95,
    "probabilities": {
        "Matang": 0.95,
        "Setengah Matang": 0.04,
        "Mentah": 0.01
    }
}
```

---

## 📊 Output Evaluasi

Setelah training, hasil evaluasi tersimpan di:

### 1. Model Tersimpan
- `output/models/best_model.h5` - Model terbaik berdasarkan validation accuracy

### 2. Visualisasi
- `output/plots/training_history.png` - Grafik akurasi dan loss per epoch
- `output/plots/confusion_matrix.png` - Confusion matrix dengan heatmap

### 3. Laporan
- `output/reports/classification_report.txt` - Precision, Recall, F1-Score per kelas

---

## 🧪 Arsitektur Model

```
Input (224x224x3)
    ↓
MobileNetV2 (Pretrained ImageNet, Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(3, Softmax)
    ↓
Output: [P(Matang), P(Setengah Matang), P(Mentah)]
```

**Training Strategy:**
- **Tahap 1**: Transfer Learning dengan base frozen (50 epoch max)
- **Tahap 2** (Opsional): Fine-tuning 20 layer terakhir dengan LR=1e-5

**Callbacks:**
- `EarlyStopping`: Hentikan jika val_loss tidak membaik (patience=10)
- `ModelCheckpoint`: Simpan model terbaik berdasarkan val_accuracy
- `ReduceLROnPlateau`: Kurangi LR jika val_loss stagnan (factor=0.5, patience=5)

---

## 📈 Target Akurasi

Untuk publikasi jurnal Sinta 2/3, target metrik:
- **Test Accuracy**: ≥ 80%
- **F1-Score per kelas**: ≥ 0.75

---

## 🔬 Data Augmentasi

Untuk meningkatkan robustitas model terhadap variasi kondisi lapangan:

- **Rotation**: ±20 derajat
- **Zoom**: ±20%
- **Horizontal Flip**: Ya
- **Brightness**: 80%-120%

---

## 📝 Lisensi Dataset

Dataset berasal dari [Roboflow](https://universe.roboflow.com/1-lwblu/tomato-0tzlu) dengan lisensi **CC BY 4.0**.

**Wajib mencantumkan atribusi** dalam publikasi jurnal:
> Dataset: Tomato Ripeness Classification, Roboflow Universe, CC BY 4.0

---

## 👨‍💻 Author

**STTR Informatika - Riset Kewirausahaan**

Untuk pertanyaan atau diskusi terkait riset, silakan hubungi dosen pembimbing.

---

## 🎯 Roadmap Deployment

1. **Fase 1**: Training dan evaluasi model (✅ Selesai)
2. **Fase 2**: Konversi ke TensorFlow Lite untuk Android
3. **Fase 3**: Integrasi ke aplikasi mobile UMKM
4. **Fase 4**: Pilot testing di lapangan dengan petani

---

## 📚 Referensi

1. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
2. Howard, A., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
3. Dataset: Roboflow Tomato Classification (CC BY 4.0)

---

**Selamat menggunakan sistem klasifikasi kematangan tomat! 🍅**
