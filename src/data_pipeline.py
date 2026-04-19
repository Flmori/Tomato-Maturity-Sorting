"""
Data Pipeline untuk preprocessing dataset klasifikasi kematangan tomat.

Modul ini menangani:
1. Pembacaan anotasi CSV dari Roboflow
2. Crop ROI (Region of Interest) dari bounding box
3. Pembuatan data generator dengan augmentasi untuk training

Author: STTR Informatika - Riset Kewirausahaan
"""

import os
import logging
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import (
    CLASS_MAPPING, 
    TARGET_SIZE, 
    DEFAULT_BATCH_SIZE,
    ensure_dir,
    get_class_label
)

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# FUNGSI PEMBACAAN ANOTASI
# ============================================================================

def load_annotations(csv_path):
    """
    Membaca file anotasi CSV dari dataset Roboflow.
    
    Format CSV yang diharapkan:
    filename,width,height,class,xmin,ymin,xmax,ymax
    
    Args:
        csv_path (str): Path ke file _annotations.csv
    
    Returns:
        pd.DataFrame: DataFrame dengan kolom yang sudah divalidasi
    
    Raises:
        FileNotFoundError: Jika file CSV tidak ditemukan
        ValueError: Jika format CSV tidak sesuai
    
    Example:
        >>> df = load_annotations('dataset/train/_annotations.csv')
        >>> print(df.head())
    """
    # Validasi file ada
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")
    
    logger.info(f"Membaca anotasi dari: {csv_path}")
    
    # Baca CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Gagal membaca CSV: {e}")
    
    # Validasi kolom yang diperlukan
    required_columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Kolom yang hilang dari CSV: {missing_columns}")
    
    # Konversi tipe data
    df['width'] = df['width'].astype(int)
    df['height'] = df['height'].astype(int)
    df['class'] = df['class'].astype(int)
    df['xmin'] = df['xmin'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    
    # Validasi nilai kelas (harus 0, 1, atau 2)
    invalid_classes = df[~df['class'].isin([0, 1, 2])]
    if not invalid_classes.empty:
        logger.warning(f"Ditemukan {len(invalid_classes)} baris dengan class tidak valid")
    
    logger.info(f"Berhasil membaca {len(df)} anotasi")
    logger.info(f"Distribusi kelas: {df['class'].value_counts().to_dict()}")
    
    return df



# ============================================================================
# FUNGSI CROP ROI
# ============================================================================

def crop_and_save_rois(annotations_df, img_dir, output_dir, target_size=TARGET_SIZE):
    """
    Crop ROI dari setiap gambar berdasarkan bounding box, resize, dan simpan.
    
    Struktur output: output_dir/{class_label}/{filename_crop}.jpg
    
    Args:
        annotations_df (pd.DataFrame): DataFrame hasil load_annotations()
        img_dir (str): Direktori gambar sumber
        output_dir (str): Direktori output terstruktur per kelas
        target_size (tuple): Ukuran target resize (default 224x224)
    
    Returns:
        dict: Statistik proses crop (jumlah berhasil, gagal, dll)
    
    Example:
        >>> df = load_annotations('dataset/train/_annotations.csv')
        >>> stats = crop_and_save_rois(df, 'dataset/train/', 'output/train/')
        >>> print(f"Berhasil: {stats['success']}, Gagal: {stats['failed']}")
    """
    logger.info(f"Memulai proses crop ROI dari {len(annotations_df)} anotasi")
    logger.info(f"Sumber: {img_dir}, Output: {output_dir}")
    
    # Statistik
    stats = {
        'success': 0,
        'failed': 0,
        'skipped_small': 0,
        'skipped_not_found': 0
    }
    
    # Buat direktori output per kelas
    for class_id, class_label in CLASS_MAPPING.items():
        class_dir = os.path.join(output_dir, class_label)
        ensure_dir(class_dir)
    
    # Proses setiap baris anotasi
    for idx, row in annotations_df.iterrows():
        try:
            # Path gambar sumber
            img_path = os.path.join(img_dir, row['filename'])
            
            # Cek apakah file ada
            if not os.path.exists(img_path):
                logger.warning(f"Gambar tidak ditemukan: {img_path}")
                stats['skipped_not_found'] += 1
                continue
            
            # Buka gambar
            img = Image.open(img_path).convert('RGB')
            
            # Crop ROI berdasarkan bounding box
            roi = img.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            
            # Validasi ukuran ROI (minimal 10x10 pixel)
            if roi.width < 10 or roi.height < 10:
                logger.warning(f"ROI terlalu kecil ({roi.width}x{roi.height}), skip: {row['filename']}")
                stats['skipped_small'] += 1
                continue
            
            # Resize ke target_size
            roi_resized = roi.resize(target_size, Image.LANCZOS)
            
            # Tentukan folder output berdasarkan kelas
            class_label = get_class_label(row['class'])
            save_dir = os.path.join(output_dir, class_label)
            
            # Simpan dengan nama unik
            base_name = os.path.splitext(row['filename'])[0]
            save_name = f"{base_name}_class{row['class']}.jpg"
            save_path = os.path.join(save_dir, save_name)
            
            roi_resized.save(save_path, quality=95)
            stats['success'] += 1
            
            # Log progress setiap 100 gambar
            if stats['success'] % 100 == 0:
                logger.info(f"Progress: {stats['success']} gambar berhasil di-crop")
        
        except Exception as e:
            logger.error(f"Error saat memproses {row['filename']}: {e}")
            stats['failed'] += 1
    
    # Log statistik akhir
    logger.info("=" * 60)
    logger.info("STATISTIK CROP ROI:")
    logger.info(f"  Berhasil: {stats['success']}")
    logger.info(f"  Gagal: {stats['failed']}")
    logger.info(f"  Skip (ROI kecil): {stats['skipped_small']}")
    logger.info(f"  Skip (file tidak ada): {stats['skipped_not_found']}")
    logger.info("=" * 60)
    
    return stats



# ============================================================================
# FUNGSI DATA GENERATOR
# ============================================================================

def create_data_generators(train_dir, valid_dir, batch_size=DEFAULT_BATCH_SIZE):
    """
    Membuat ImageDataGenerator dengan augmentasi untuk training dan validasi.
    
    Augmentasi untuk training:
    - Rotation: ±20 derajat
    - Zoom: ±20%
    - Horizontal flip
    - Brightness: 80%-120%
    
    Validasi hanya menggunakan rescaling tanpa augmentasi.
    
    Args:
        train_dir (str): Direktori data training (harus berisi subfolder per kelas)
        valid_dir (str): Direktori data validasi (harus berisi subfolder per kelas)
        batch_size (int): Ukuran batch (default: 32)
    
    Returns:
        tuple: (train_generator, valid_generator)
    
    Example:
        >>> train_gen, valid_gen = create_data_generators(
        ...     'output/train/', 
        ...     'output/valid/'
        ... )
        >>> print(f"Training batches: {len(train_gen)}")
    """
    logger.info("Membuat data generators...")
    
    # Generator untuk training dengan augmentasi
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalisasi ke [0, 1]
        rotation_range=20,           # Rotasi ±20 derajat
        zoom_range=0.2,              # Zoom ±20%
        horizontal_flip=True,        # Flip horizontal
        brightness_range=[0.8, 1.2], # Brightness 80%-120%
        fill_mode='nearest'          # Mode fill untuk pixel kosong
    )
    
    # Generator untuk validasi tanpa augmentasi
    valid_datagen = ImageDataGenerator(
        rescale=1./255  # Hanya normalisasi
    )
    
    # Buat generator dari direktori
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical',  # One-hot encoding untuk 3 kelas
        shuffle=True,
        seed=42
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Tidak shuffle untuk validasi
    )
    
    # Log informasi generator
    logger.info(f"Training generator:")
    logger.info(f"  - Total samples: {train_generator.samples}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Steps per epoch: {len(train_generator)}")
    logger.info(f"  - Classes: {train_generator.class_indices}")
    
    logger.info(f"Validation generator:")
    logger.info(f"  - Total samples: {valid_generator.samples}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Steps per epoch: {len(valid_generator)}")
    
    return train_generator, valid_generator



def create_test_generator(test_dir, batch_size=DEFAULT_BATCH_SIZE):
    """
    Membuat generator untuk test set (tanpa augmentasi, shuffle=False).
    
    Generator ini digunakan untuk evaluasi final model.
    Tidak ada augmentasi dan tidak di-shuffle untuk hasil yang reproducible.
    
    Args:
        test_dir (str): Direktori data test (harus berisi subfolder per kelas)
        batch_size (int): Ukuran batch (default: 32)
    
    Returns:
        DirectoryIterator: Test generator
    
    Example:
        >>> test_gen = create_test_generator('output/test/')
        >>> print(f"Test samples: {test_gen.samples}")
    """
    logger.info("Membuat test generator...")
    
    # Generator tanpa augmentasi
    test_datagen = ImageDataGenerator(
        rescale=1./255  # Hanya normalisasi
    )
    
    # Buat generator dari direktori
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # PENTING: Tidak shuffle untuk evaluasi
    )
    
    logger.info(f"Test generator:")
    logger.info(f"  - Total samples: {test_generator.samples}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Steps: {len(test_generator)}")
    logger.info(f"  - Classes: {test_generator.class_indices}")
    
    return test_generator
