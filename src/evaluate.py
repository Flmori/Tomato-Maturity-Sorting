"""
Evaluasi dan visualisasi untuk model klasifikasi kematangan tomat.

Modul ini menangani:
1. Plotting training history (akurasi dan loss)
2. Confusion matrix
3. Classification report (Precision, Recall, F1-Score)

Author: STTR Informatika - Riset Kewirausahaan
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from utils import CLASS_NAMES, ensure_dir

# Setup logging
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# FUNGSI PLOTTING TRAINING HISTORY
# ============================================================================

def plot_training_history(history, save_path):
    """
    Plot grafik akurasi dan loss (training vs validasi) dalam satu figure.
    
    Args:
        history: Keras History object dari model.fit()
        save_path (str): Path untuk menyimpan grafik (PNG)
    
    Example:
        >>> history = model.fit(...)
        >>> plot_training_history(history, 'output/plots/training_history.png')
    """
    logger.info(f"Membuat plot training history...")
    
    # Pastikan direktori output ada
    output_dir = os.path.dirname(save_path)
    ensure_dir(output_dir)
    
    # Buat figure dengan 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Akurasi
    axes[0].plot(history.history['accuracy'], label='Training', color='blue', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validasi', color='orange', linewidth=2)
    axes[0].set_title('Akurasi Model per Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Akurasi', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[1].plot(history.history['loss'], label='Training', color='blue', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validasi', color='orange', linewidth=2)
    axes[1].set_title('Loss Model per Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot disimpan ke: {save_path}")



# ============================================================================
# FUNGSI CONFUSION MATRIX
# ============================================================================

def generate_confusion_matrix(model, test_gen, class_names=CLASS_NAMES, save_path=None):
    """
    Generate dan visualisasi confusion matrix menggunakan seaborn heatmap.
    
    Args:
        model: Model Keras yang sudah di-train
        test_gen: Test data generator
        class_names (list): List nama kelas (default: ["Matang", "Setengah Matang", "Mentah"])
        save_path (str): Path untuk menyimpan grafik (optional)
    
    Returns:
        np.ndarray: Confusion matrix
    
    Example:
        >>> cm = generate_confusion_matrix(
        ...     model, 
        ...     test_gen, 
        ...     save_path='output/plots/confusion_matrix.png'
        ... )
    """
    logger.info("Membuat confusion matrix...")
    
    # Prediksi pada test set
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Ground truth labels
    y_true = test_gen.classes
    
    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Hitung persentase per baris
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Buat heatmap
    plt.figure(figsize=(10, 8))
    
    # Annotasi dengan jumlah absolut dan persentase
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Jumlah Prediksi'},
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title('Confusion Matrix - Klasifikasi Kematangan Tomat', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Prediksi', fontsize=12, fontweight='bold')
    plt.ylabel('Label Sebenarnya', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix disimpan ke: {save_path}")
    
    plt.close()
    
    return cm



# ============================================================================
# FUNGSI CLASSIFICATION REPORT
# ============================================================================

def generate_classification_report(model, test_gen, class_names=CLASS_NAMES, save_path=None):
    """
    Generate classification report (Precision, Recall, F1-Score per kelas).
    
    Args:
        model: Model Keras yang sudah di-train
        test_gen: Test data generator
        class_names (list): List nama kelas
        save_path (str): Path untuk menyimpan report sebagai file teks (optional)
    
    Returns:
        dict: Classification report sebagai dictionary
    
    Example:
        >>> report = generate_classification_report(
        ...     model, 
        ...     test_gen,
        ...     save_path='output/reports/classification_report.txt'
        ... )
        >>> print(f"Accuracy: {report['accuracy']:.4f}")
    """
    logger.info("Membuat classification report...")
    
    # Prediksi pada test set
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Ground truth labels
    y_true = test_gen.classes
    
    # Hitung akurasi keseluruhan
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    report_str = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4
    )
    
    # Parse report ke dictionary
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Log report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    logger.info(f"\n{report_str}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info("=" * 60)
    
    # Simpan ke file jika diminta
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("Sistem Klasifikasi Kematangan Tomat\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_str)
            f.write(f"\n\nOverall Accuracy: {accuracy:.4f}\n")
            f.write(f"Total Test Samples: {len(y_true)}\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Classification report disimpan ke: {save_path}")
    
    return report_dict



# ============================================================================
# FUNGSI EVALUASI LENGKAP
# ============================================================================

def evaluate_model(model, test_gen, class_names=CLASS_NAMES, output_dir='output'):
    """
    Fungsi utama evaluasi: memanggil semua fungsi evaluasi di atas
    dan menyimpan semua output ke output_dir.
    
    Args:
        model: Model Keras yang sudah di-train
        test_gen: Test data generator
        class_names (list): List nama kelas
        output_dir (str): Direktori output untuk plots dan reports
    
    Returns:
        dict: Dictionary berisi semua hasil evaluasi
    
    Example:
        >>> results = evaluate_model(model, test_gen, output_dir='output')
        >>> print(f"Test Accuracy: {results['accuracy']:.4f}")
    """
    logger.info("=" * 60)
    logger.info("MEMULAI EVALUASI MODEL")
    logger.info("=" * 60)
    
    # Path untuk output
    plots_dir = os.path.join(output_dir, 'plots')
    reports_dir = os.path.join(output_dir, 'reports')
    
    ensure_dir(plots_dir)
    ensure_dir(reports_dir)
    
    # 1. Generate confusion matrix
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    cm = generate_confusion_matrix(model, test_gen, class_names, save_path=cm_path)
    
    # 2. Generate classification report
    report_path = os.path.join(reports_dir, 'classification_report.txt')
    report = generate_classification_report(model, test_gen, class_names, save_path=report_path)
    
    # 3. Hitung akurasi test set
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    test_accuracy = accuracy_score(y_true, y_pred)
    
    logger.info("=" * 60)
    logger.info("EVALUASI SELESAI")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Confusion Matrix: {cm_path}")
    logger.info(f"Classification Report: {report_path}")
    
    # Kembalikan hasil
    results = {
        'accuracy': test_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'true_labels': y_true
    }
    
    return results
