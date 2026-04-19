"""
Entry point utama untuk sistem klasifikasi kematangan tomat.

Script ini menjalankan full pipeline:
1. Preprocessing: Crop ROI dari bounding box
2. Training: Transfer Learning MobileNetV2
3. Evaluasi: Confusion Matrix, Classification Report, Grafik

Author: STTR Informatika - Riset Kewirausahaan
Judul: Otomasi Grading Kematangan Tomat Berbasis Deep Learning untuk Efisiensi Operasional UMKM Pertanian
"""

import os
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np

# Tambahkan src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import setup_logging, RANDOM_SEED
from data_pipeline import (
    load_annotations,
    crop_and_save_rois,
    create_data_generators,
    create_test_generator
)
from model import build_mobilenetv2_model, unfreeze_top_layers
from train import get_callbacks, train_model, fine_tune_model
from evaluate import plot_training_history, evaluate_model
from inference import grading_tomat

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def set_random_seeds():
    """Set random seeds untuk reproducibility."""
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    logger.info(f"Random seed set to: {RANDOM_SEED}")


def run_preprocessing(args):
    """
    Jalankan preprocessing: crop ROI dari bounding box.
    
    Args:
        args: Argparse arguments
    """
    logger.info("=" * 60)
    logger.info("TAHAP 1: PREPROCESSING DATA")
    logger.info("=" * 60)
    
    # Proses train set
    logger.info("Processing TRAIN set...")
    train_df = load_annotations('dataset/train/_annotations.csv')
    train_stats = crop_and_save_rois(
        train_df, 
        'dataset/train/', 
        'output/cropped/train/'
    )
    
    # Proses valid set
    logger.info("Processing VALID set...")
    valid_df = load_annotations('dataset/valid/_annotations.csv')
    valid_stats = crop_and_save_rois(
        valid_df, 
        'dataset/valid/', 
        'output/cropped/valid/'
    )
    
    # Proses test set
    logger.info("Processing TEST set...")
    test_df = load_annotations('dataset/test/_annotations.csv')
    test_stats = crop_and_save_rois(
        test_df, 
        'dataset/test/', 
        'output/cropped/test/'
    )
    
    logger.info("Preprocessing selesai!")
    logger.info(f"Total berhasil: {train_stats['success'] + valid_stats['success'] + test_stats['success']}")


def run_training(args):
    """
    Jalankan training model.
    
    Args:
        args: Argparse arguments
    """
    logger.info("=" * 60)
    logger.info("TAHAP 2: TRAINING MODEL")
    logger.info("=" * 60)
    
    # Buat data generators
    train_gen, valid_gen = create_data_generators(
        'output/cropped/train/',
        'output/cropped/valid/',
        batch_size=args.batch_size
    )
    
    # Build model
    model = build_mobilenetv2_model(
        learning_rate=args.learning_rate
    )
    
    # Get callbacks
    callbacks = get_callbacks('output/models/best_model.h5')
    
    # Training
    history = train_model(
        model,
        train_gen,
        valid_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history, 'output/plots/training_history.png')
    
    logger.info("Training selesai!")
    
    return model, history


def run_evaluation(args):
    """
    Jalankan evaluasi model pada test set.
    
    Args:
        args: Argparse arguments
    """
    logger.info("=" * 60)
    logger.info("TAHAP 3: EVALUASI MODEL")
    logger.info("=" * 60)
    
    # Load model terbaik
    model = tf.keras.models.load_model('output/models/best_model.h5')
    
    # Buat test generator
    test_gen = create_test_generator(
        'output/cropped/test/',
        batch_size=args.batch_size
    )
    
    # Evaluasi
    results = evaluate_model(model, test_gen, output_dir='output')
    
    logger.info("Evaluasi selesai!")
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    
    return results


def main():
    """Fungsi utama."""
    parser = argparse.ArgumentParser(
        description='Sistem Klasifikasi Kematangan Tomat - Riset Kewirausahaan STTR'
    )
    
    # Arguments
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip tahap preprocessing (jika ROI sudah di-crop)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip tahap training (jika model sudah ada)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip tahap evaluasi'
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Aktifkan fine-tuning setelah training awal'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Jumlah epoch untuk training (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds()
    
    logger.info("=" * 60)
    logger.info("SISTEM KLASIFIKASI KEMATANGAN TOMAT")
    logger.info("Riset Kewirausahaan - STTR Informatika")
    logger.info("=" * 60)
    
    # Jalankan pipeline
    if not args.skip_preprocessing:
        run_preprocessing(args)
    
    if not args.skip_training:
        model, history = run_training(args)
        
        # Fine-tuning opsional
        if args.fine_tune:
            logger.info("Memulai fine-tuning...")
            model = unfreeze_top_layers(model, num_layers=20)
            callbacks = get_callbacks('output/models/best_model_finetuned.h5')
            history_ft = fine_tune_model(
                model,
                train_gen,
                valid_gen,
                epochs=20,
                callbacks=callbacks
            )
    
    if not args.skip_evaluation:
        results = run_evaluation(args)
    
    logger.info("=" * 60)
    logger.info("PIPELINE SELESAI!")
    logger.info("=" * 60)
    logger.info("Output:")
    logger.info("  - Model: output/models/best_model.h5")
    logger.info("  - Plots: output/plots/")
    logger.info("  - Reports: output/reports/")


if __name__ == '__main__':
    main()
