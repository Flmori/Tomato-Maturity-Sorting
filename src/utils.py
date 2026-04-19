"""
Utility functions dan konstanta global untuk sistem klasifikasi kematangan tomat.

Modul ini berisi:
- Mapping kelas (integer ke label string)
- Konstanta input shape untuk MobileNetV2
- Helper functions untuk logging dan validasi

Author: STTR Informatika - Riset Kewirausahaan
"""

import os
import logging

# ============================================================================
# KONSTANTA GLOBAL
# ============================================================================

# Mapping kelas dari integer ke label Bahasa Indonesia
CLASS_MAPPING = {
    0: "Matang",           # Ripe - tomat merah penuh
    1: "Setengah Matang",  # Half-ripe - tomat oranye/merah muda
    2: "Mentah"            # Unripe - tomat hijau
}

# List nama kelas untuk keperluan plotting dan evaluasi
CLASS_NAMES = ["Matang", "Setengah Matang", "Mentah"]

# Input shape untuk MobileNetV2 (height, width, channels)
INPUT_SHAPE = (224, 224, 3)

# Target size untuk resize gambar
TARGET_SIZE = (224, 224)

# Batch size default untuk training
DEFAULT_BATCH_SIZE = 32

# Learning rate default
DEFAULT_LEARNING_RATE = 0.001

# Jumlah kelas
NUM_CLASSES = 3

# Random seed untuk reproducibility
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_logging(log_level=logging.INFO):
    """
    Setup konfigurasi logging untuk seluruh aplikasi.
    
    Args:
        log_level: Level logging (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_dir(directory):
    """
    Pastikan direktori ada, jika tidak maka buat.
    
    Args:
        directory: Path direktori yang akan dibuat
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Direktori dibuat: {directory}")


def get_class_label(class_id):
    """
    Konversi class ID (integer) ke label string.
    
    Args:
        class_id: Integer class ID (0, 1, atau 2)
    
    Returns:
        String label kelas
    
    Raises:
        ValueError: Jika class_id tidak valid
    """
    if class_id not in CLASS_MAPPING:
        raise ValueError(f"Class ID tidak valid: {class_id}. Harus 0, 1, atau 2.")
    return CLASS_MAPPING[class_id]


def validate_image_path(image_path):
    """
    Validasi apakah path gambar valid dan file ada.
    
    Args:
        image_path: Path ke file gambar
    
    Returns:
        True jika valid
    
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
        ValueError: Jika format file tidak didukung
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")
    
    # Cek ekstensi file
    valid_extensions = ['.jpg', '.jpeg', '.png']
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in valid_extensions:
        raise ValueError(f"Format file tidak didukung: {ext}. Gunakan JPG atau PNG.")
    
    return True
