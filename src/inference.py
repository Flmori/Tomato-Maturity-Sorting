"""
Inferensi untuk prediksi kematangan tomat pada gambar baru.

Modul ini menyediakan:
1. Load model dengan caching
2. Preprocessing gambar untuk inferensi
3. Fungsi grading_tomat() untuk prediksi single image

Author: STTR Informatika - Riset Kewirausahaan
"""

import logging
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from utils import (
    CLASS_NAMES,
    TARGET_SIZE,
    validate_image_path
)

# Setup logging
logger = logging.getLogger(__name__)

# Cache untuk model (agar tidak perlu load berulang kali)
_model_cache = {}


# ============================================================================
# FUNGSI LOAD MODEL
# ============================================================================

def load_model(model_path):
    """
    Load model dari file .h5 atau SavedModel directory dengan caching.
    
    Model di-cache setelah load pertama untuk efisiensi inferensi berulang.
    
    Args:
        model_path (str): Path ke model (.h5 atau direktori SavedModel)
    
    Returns:
        tf.keras.Model: Model yang sudah di-load
    
    Raises:
        FileNotFoundError: Jika model tidak ditemukan
    
    Example:
        >>> model = load_model('output/models/best_model.h5')
        >>> # Load kedua akan menggunakan cache
        >>> model = load_model('output/models/best_model.h5')
    """
    # Cek apakah model sudah ada di cache
    if model_path in _model_cache:
        logger.info(f"Menggunakan cached model: {model_path}")
        return _model_cache[model_path]
    
    # Validasi file ada
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            f"Pastikan Anda sudah menjalankan training terlebih dahulu."
        )
    
    logger.info(f"Loading model dari: {model_path}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Simpan ke cache
        _model_cache[model_path] = model
        
        logger.info("Model berhasil di-load")
        return model
    
    except Exception as e:
        raise ValueError(f"Gagal load model: {e}")



# ============================================================================
# FUNGSI PREPROCESSING
# ============================================================================

def preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Preprocessing gambar untuk inferensi.
    
    Langkah preprocessing:
    1. Load gambar (RGB)
    2. Resize ke target_size
    3. Normalisasi ke [0, 1]
    4. Expand dims untuk batch dimension
    
    Args:
        image_path (str): Path ke gambar (JPG/PNG)
        target_size (tuple): Ukuran target (default: (224, 224))
    
    Returns:
        np.ndarray: Array dengan shape (1, 224, 224, 3)
    
    Example:
        >>> img_array = preprocess_image('test_image.jpg')
        >>> print(img_array.shape)  # (1, 224, 224, 3)
    """
    # Validasi path gambar
    validate_image_path(image_path)
    
    # Load gambar
    img = Image.open(image_path).convert('RGB')
    
    # Resize ke target_size
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    # Konversi ke numpy array
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Normalisasi ke [0, 1]
    img_array = img_array / 255.0
    
    # Expand dims untuk batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array



# ============================================================================
# FUNGSI GRADING TOMAT
# ============================================================================

def grading_tomat(image_path, model_path='output/models/best_model.h5'):
    """
    Fungsi utama grading kematangan tomat.
    
    Args:
        image_path (str): Path ke gambar tomat (JPG/PNG, ukuran bebas)
        model_path (str): Path ke model tersimpan (default: output/models/best_model.h5)
    
    Returns:
        dict: {
            "label": str,           # "Matang" | "Setengah Matang" | "Mentah"
            "confidence": float,    # 0.0 - 1.0
            "probabilities": {      # Probabilitas semua kelas
                "Matang": float,
                "Setengah Matang": float,
                "Mentah": float
            }
        }
    
    Example:
        >>> result = grading_tomat('test_tomato.jpg')
        >>> print(f"Kematangan: {result['label']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    logger.info(f"Memulai grading untuk: {image_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Preprocessing gambar
    img_array = preprocess_image(image_path)
    
    # Prediksi
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]  # shape (3,)
    
    # Tentukan kelas prediksi
    predicted_idx = np.argmax(probs)
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    # Susun output
    result = {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[0]: float(probs[0]),  # Matang
            CLASS_NAMES[1]: float(probs[1]),  # Setengah Matang
            CLASS_NAMES[2]: float(probs[2])   # Mentah
        }
    }
    
    logger.info(f"Hasil prediksi: {predicted_label} (confidence: {confidence:.2%})")
    
    return result
