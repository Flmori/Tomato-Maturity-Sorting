"""
Arsitektur model untuk klasifikasi kematangan tomat menggunakan Transfer Learning.

Modul ini mengimplementasikan:
1. MobileNetV2 sebagai base model (pretrained ImageNet)
2. Custom classification head
3. Fine-tuning utilities

Author: STTR Informatika - Riset Kewirausahaan
"""

import logging
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

from utils import (
    NUM_CLASSES,
    INPUT_SHAPE,
    DEFAULT_LEARNING_RATE
)

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# FUNGSI BUILD MODEL
# ============================================================================

def build_mobilenetv2_model(
    num_classes=NUM_CLASSES,
    input_shape=INPUT_SHAPE,
    learning_rate=DEFAULT_LEARNING_RATE
):
    """
    Membangun model MobileNetV2 dengan custom head untuk klasifikasi 3 kelas.
    
    Arsitektur:
    - Base: MobileNetV2 pretrained ImageNet (frozen)
    - Custom Head: GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(3, softmax)
    
    Args:
        num_classes (int): Jumlah kelas output (default: 3)
        input_shape (tuple): Shape input gambar (default: (224, 224, 3))
        learning_rate (float): Learning rate untuk optimizer (default: 0.001)
    
    Returns:
        tf.keras.Model: Model yang sudah dikompilasi
    
    Example:
        >>> model = build_mobilenetv2_model()
        >>> model.summary()
    """
    logger.info("Membangun model MobileNetV2...")
    logger.info(f"  - Input shape: {input_shape}")
    logger.info(f"  - Num classes: {num_classes}")
    logger.info(f"  - Learning rate: {learning_rate}")
    
    # Load MobileNetV2 pretrained ImageNet tanpa top layer
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Tidak pakai classification head bawaan
        weights='imagenet'  # Gunakan pretrained weights
    )
    
    # Freeze semua layer base (Transfer Learning tahap 1)
    base_model.trainable = False
    logger.info(f"Base model loaded: {len(base_model.layers)} layers (frozen)")
    
    # Bangun custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.5, name='dropout')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Buat model final
    model = Model(inputs=base_model.input, outputs=output, name='tomato_classifier')
    
    # Kompilasi model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model berhasil dibuat dan dikompilasi")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    # Hitung trainable vs non-trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    logger.info(f"  - Trainable: {trainable_params:,}")
    logger.info(f"  - Non-trainable: {non_trainable_params:,}")
    
    return model



# ============================================================================
# FUNGSI FINE-TUNING
# ============================================================================

def unfreeze_top_layers(model, num_layers=20):
    """
    Unfreeze N layer terakhir dari base MobileNetV2 untuk fine-tuning tahap 2.
    
    Fine-tuning dilakukan dengan learning rate yang lebih kecil (1e-5)
    untuk mencegah catastrophic forgetting.
    
    Args:
        model (tf.keras.Model): Model hasil build_mobilenetv2_model()
        num_layers (int): Jumlah layer dari akhir yang di-unfreeze (default: 20)
    
    Returns:
        tf.keras.Model: Model dengan layer terakhir trainable
    
    Example:
        >>> model = build_mobilenetv2_model()
        >>> # Training tahap 1 selesai...
        >>> model = unfreeze_top_layers(model, num_layers=20)
        >>> # Lanjutkan training dengan learning rate kecil
    """
    logger.info(f"Unfreeze {num_layers} layer terakhir untuk fine-tuning...")
    
    # Cari base model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        # Fallback: cari layer pertama yang merupakan Model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
    
    if base_model is None:
        raise ValueError("Base model (MobileNetV2) tidak ditemukan dalam model")
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze semua layer kecuali N layer terakhir
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    # Recompile dengan learning rate lebih kecil
    fine_tune_lr = 1e-5
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Hitung trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    logger.info(f"Fine-tuning configuration:")
    logger.info(f"  - Total layers in base: {total_layers}")
    logger.info(f"  - Frozen layers: {freeze_until}")
    logger.info(f"  - Trainable layers: {num_layers}")
    logger.info(f"  - Learning rate: {fine_tune_lr}")
    logger.info(f"  - Trainable params: {trainable_params:,}")
    logger.info(f"  - Non-trainable params: {non_trainable_params:,}")
    
    return model
