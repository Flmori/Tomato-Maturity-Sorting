"""
Training utilities untuk model klasifikasi kematangan tomat.

Modul ini menangani:
1. Callbacks untuk training (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
2. Training loop utama
3. Fine-tuning opsional

Author: STTR Informatika - Riset Kewirausahaan
"""

import logging
import os
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from utils import ensure_dir

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# FUNGSI CALLBACKS
# ============================================================================

def get_callbacks(model_save_path):
    """
    Membuat list callbacks untuk training.
    
    Callbacks yang digunakan:
    1. EarlyStopping: Hentikan training jika val_loss tidak membaik
    2. ModelCheckpoint: Simpan model terbaik berdasarkan val_accuracy
    3. ReduceLROnPlateau: Kurangi learning rate jika val_loss stagnan
    
    Args:
        model_save_path (str): Path untuk menyimpan model terbaik
    
    Returns:
        list: List of Keras callbacks
    
    Example:
        >>> callbacks = get_callbacks('output/models/best_model.h5')
        >>> history = model.fit(..., callbacks=callbacks)
    """
    logger.info("Membuat callbacks untuk training...")
    
    # Pastikan direktori output ada
    model_dir = os.path.dirname(model_save_path)
    ensure_dir(model_dir)
    
    # 1. EarlyStopping: Hentikan training jika tidak ada improvement
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    logger.info("  - EarlyStopping: monitor=val_loss, patience=10")
    
    # 2. ModelCheckpoint: Simpan model terbaik
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    logger.info(f"  - ModelCheckpoint: save to {model_save_path}")
    
    # 3. ReduceLROnPlateau: Kurangi learning rate jika stagnan
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    logger.info("  - ReduceLROnPlateau: factor=0.5, patience=5")
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    return callbacks



# ============================================================================
# FUNGSI TRAINING
# ============================================================================

def train_model(model, train_gen, valid_gen, epochs=50, callbacks=None):
    """
    Menjalankan training loop utama (Transfer Learning tahap 1).
    
    Args:
        model (tf.keras.Model): Model yang akan di-train
        train_gen: Training data generator
        valid_gen: Validation data generator
        epochs (int): Maksimal jumlah epoch (default: 50)
        callbacks (list): List of callbacks (optional)
    
    Returns:
        History: Keras History object untuk plotting
    
    Example:
        >>> model = build_mobilenetv2_model()
        >>> callbacks = get_callbacks('output/models/best_model.h5')
        >>> history = train_model(model, train_gen, valid_gen, epochs=50, callbacks=callbacks)
    """
    logger.info("=" * 60)
    logger.info("MEMULAI TRAINING (Transfer Learning Tahap 1)")
    logger.info("=" * 60)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Training samples: {train_gen.samples}")
    logger.info(f"Validation samples: {valid_gen.samples}")
    logger.info(f"Steps per epoch: {len(train_gen)}")
    
    # Training
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING SELESAI")
    logger.info("=" * 60)
    
    # Log hasil akhir
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
    logger.info(f"Final Training Loss: {final_train_loss:.4f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    
    return history



def fine_tune_model(model, train_gen, valid_gen, epochs=20, callbacks=None):
    """
    Fine-tuning opsional: unfreeze top layers dan lanjutkan training
    dengan learning rate lebih kecil (1e-5).
    
    Fungsi ini harus dipanggil SETELAH training tahap 1 selesai.
    Model sudah harus di-unfreeze menggunakan unfreeze_top_layers().
    
    Args:
        model (tf.keras.Model): Model yang sudah di-unfreeze
        train_gen: Training data generator
        valid_gen: Validation data generator
        epochs (int): Jumlah epoch untuk fine-tuning (default: 20)
        callbacks (list): List of callbacks (optional)
    
    Returns:
        History: Keras History object untuk plotting
    
    Example:
        >>> # Setelah training tahap 1
        >>> model = unfreeze_top_layers(model, num_layers=20)
        >>> callbacks = get_callbacks('output/models/best_model_finetuned.h5')
        >>> history_ft = fine_tune_model(model, train_gen, valid_gen, epochs=20, callbacks=callbacks)
    """
    logger.info("=" * 60)
    logger.info("MEMULAI FINE-TUNING (Transfer Learning Tahap 2)")
    logger.info("=" * 60)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Training samples: {train_gen.samples}")
    logger.info(f"Validation samples: {valid_gen.samples}")
    
    # Fine-tuning
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("=" * 60)
    logger.info("FINE-TUNING SELESAI")
    logger.info("=" * 60)
    
    # Log hasil akhir
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
    logger.info(f"Final Training Loss: {final_train_loss:.4f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    
    return history
