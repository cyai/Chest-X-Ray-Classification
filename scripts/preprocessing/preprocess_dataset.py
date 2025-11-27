#!/usr/bin/env python3
"""
Dataset Preprocessing and Standardization Script
Fixes irregularities in chest X-ray dataset and creates a clean, standardized version

Usage:
    python preprocess_dataset.py --input chest_xray_dataset --output chest_xray_clean
    python preprocess_dataset.py --input chest_xray_dataset --output chest_xray_clean --remove-outliers
    python preprocess_dataset.py --input chest_xray_dataset --output chest_xray_clean --size 512 --apply-clahe
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

warnings.filterwarnings('ignore')


class DatasetPreprocessor:
    """Comprehensive dataset preprocessing and standardization"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        target_size: int = 224,
        apply_clahe: bool = True,
        convert_to_grayscale: bool = True,
        remove_outliers: bool = False,
        outlier_list: str = None,
        quality_threshold: float = 0.95
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.convert_to_grayscale = convert_to_grayscale
        self.remove_outliers = remove_outliers
        self.outlier_list = outlier_list
        self.quality_threshold = quality_threshold
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'outliers_removed': 0,
            'by_class': defaultdict(lambda: {
                'processed': 0,
                'skipped': 0,
                'errors': 0
            })
        }
        
        # Outlier set
        self.outliers_to_remove: Set[str] = set()
        
        # CLAHE object
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        print("🔧 Dataset Preprocessor initialized")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Target size: {self.target_size}x{self.target_size}")
        print(f"   Convert to grayscale: {self.convert_to_grayscale}")
        print(f"   Apply CLAHE: {self.apply_clahe}")
        print(f"   Remove outliers: {self.remove_outliers}")
    
    def load_outlier_list(self):
        """Load list of outlier images to remove"""
        if not self.outlier_list:
            return
        
        print(f"\n📋 Loading outlier list from {self.outlier_list}...")
        
        try:
            df = pd.read_csv(self.outlier_list)
            # Filter for severe outliers (3+ anomalies)
            severe = df[df['outlier_count'] >= 3]
            
            for _, row in severe.iterrows():
                self.outliers_to_remove.add(row['filename'])
            
            print(f"   Loaded {len(self.outliers_to_remove)} outlier filenames to remove")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load outlier list: {e}")
    
    def should_skip_image(self, filename: str) -> bool:
        """Check if image should be skipped"""
        if self.remove_outliers and filename in self.outliers_to_remove:
            return True
        return False
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to a single image
        
        Steps:
        1. Convert to grayscale (if needed)
        2. Resize to target size
        3. Apply CLAHE normalization (if enabled)
        4. Normalize pixel values
        
        Args:
            img: Input image (H, W, C) or (H, W)
            
        Returns:
            Preprocessed image
        """
        # Step 1: Convert to grayscale
        if self.convert_to_grayscale:
            if len(img.shape) == 3:
                # If RGB, convert to grayscale
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            # Ensure RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Step 2: Resize with aspect ratio preservation (pad if needed)
        img = self.resize_with_padding(img, self.target_size)
        
        # Step 3: Apply CLAHE for brightness/contrast normalization
        if self.apply_clahe:
            if len(img.shape) == 2:
                # Grayscale
                img = self.clahe.apply(img)
            else:
                # RGB - apply to each channel
                img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                img_yuv[:, :, 0] = self.clahe.apply(img_yuv[:, :, 0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        return img
    
    def resize_with_padding(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image to target size while preserving aspect ratio
        Pads with black pixels if needed
        """
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image
        if len(img.shape) == 2:
            padded = np.zeros((target_size, target_size), dtype=np.uint8)
        else:
            padded = np.zeros((target_size, target_size, img.shape[2]), dtype=np.uint8)
        
        # Calculate padding
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        
        # Place resized image in center
        padded[top:top+new_h, left:left+new_w] = img
        
        return padded
    
    def process_image(
        self,
        input_path: Path,
        output_path: Path,
        class_name: str
    ) -> bool:
        """
        Process a single image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if should skip
            if self.should_skip_image(input_path.name):
                self.stats['total_skipped'] += 1
                self.stats['by_class'][class_name]['skipped'] += 1
                self.stats['outliers_removed'] += 1
                return False
            
            # Load image
            img = cv2.imread(str(input_path))
            
            if img is None:
                raise ValueError(f"Failed to load image: {input_path}")
            
            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            processed = self.preprocess_image(img)
            
            # Convert back to BGR for saving
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            
            # Save with high quality
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(output_path),
                processed,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['by_class'][class_name]['processed'] += 1
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {input_path.name}: {e}")
            self.stats['total_errors'] += 1
            self.stats['by_class'][class_name]['errors'] += 1
            return False
    
    def process_split(self, split_name: str):
        """Process all images in a split (train/val/test)"""
        split_dir = self.input_dir / split_name
        
        if not split_dir.exists():
            print(f"   ⚠️  Split directory not found: {split_dir}")
            return
        
        print(f"\n📁 Processing {split_name} split...")
        
        # Get all class directories
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            print(f"   Processing class: {class_name}")
            
            # Get all images
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            # Process with progress bar
            for img_path in tqdm(image_files, desc=f"   {class_name}", ncols=100):
                output_path = self.output_dir / split_name / class_name / img_path.name
                self.process_image(img_path, output_path, class_name)
    
    def generate_comparison_samples(self, num_samples: int = 5):
        """Generate before/after comparison images"""
        print(f"\n🖼️  Generating {num_samples} before/after comparison samples...")
        
        comparison_dir = self.output_dir / 'comparison_samples'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        splits = ['train', 'val', 'test']
        classes = ['normal', 'pneumonia', 'tuberculosis']
        
        for class_name in classes:
            print(f"   Creating samples for {class_name}...")
            
            # Find images from any split
            for split in splits:
                input_class_dir = self.input_dir / split / class_name
                output_class_dir = self.output_dir / split / class_name
                
                if not input_class_dir.exists():
                    continue
                
                # Get sample images
                images = list(input_class_dir.glob('*.jpg'))[:num_samples]
                
                for idx, img_path in enumerate(images):
                    try:
                        # Load original
                        original = cv2.imread(str(img_path))
                        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                        
                        # Load processed
                        processed_path = output_class_dir / img_path.name
                        if not processed_path.exists():
                            continue
                        
                        processed = cv2.imread(str(processed_path))
                        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                        
                        # Resize original to match for comparison
                        original_resized = cv2.resize(
                            original_rgb,
                            (self.target_size, self.target_size),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                        
                        # Create side-by-side comparison
                        comparison = np.hstack([original_resized, processed_rgb])
                        
                        # Add labels
                        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            comparison_bgr,
                            'Original',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            comparison_bgr,
                            'Processed',
                            (self.target_size + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        # Save
                        output_file = comparison_dir / f"{class_name}_{idx+1}_comparison.jpg"
                        cv2.imwrite(str(output_file), comparison_bgr)
                        
                    except Exception as e:
                        print(f"      ⚠️  Error creating comparison for {img_path.name}: {e}")
                
                break  # Only use first split that has images
        
        print(f"   ✅ Comparison samples saved to {comparison_dir}")
    
    def save_statistics(self):
        """Save processing statistics"""
        stats_file = self.output_dir / 'preprocessing_stats.json'
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n📊 Statistics saved to {stats_file}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("📊 PREPROCESSING SUMMARY")
        print("="*70)
        print(f"\n✅ Total images processed: {self.stats['total_processed']}")
        print(f"⏭️  Total images skipped: {self.stats['total_skipped']}")
        if self.remove_outliers:
            print(f"🗑️  Outliers removed: {self.stats['outliers_removed']}")
        print(f"❌ Total errors: {self.stats['total_errors']}")
        
        print("\n📦 Per-Class Summary:")
        print("-" * 70)
        print(f"{'Class':<15} {'Processed':<12} {'Skipped':<12} {'Errors':<12}")
        print("-" * 70)
        
        for class_name, class_stats in sorted(self.stats['by_class'].items()):
            print(
                f"{class_name:<15} "
                f"{class_stats['processed']:<12} "
                f"{class_stats['skipped']:<12} "
                f"{class_stats['errors']:<12}"
            )
        
        print("-" * 70)
        
        success_rate = (
            self.stats['total_processed'] / 
            (self.stats['total_processed'] + self.stats['total_errors']) * 100
            if (self.stats['total_processed'] + self.stats['total_errors']) > 0
            else 0
        )
        
        print(f"\n✨ Success rate: {success_rate:.2f}%")
        print(f"\n📁 Cleaned dataset saved to: {self.output_dir}")
        print("="*70)
    
    def create_yaml_config(self):
        """Create data.yaml config file for the cleaned dataset"""
        yaml_content = f"""# Cleaned Chest X-Ray Dataset Configuration
# Preprocessed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

# Dataset paths
train: train
val: val
test: test

# Class information
nc: 3  # number of classes
names: ['normal', 'pneumonia', 'tuberculosis']

# Preprocessing applied
preprocessing:
  target_size: {self.target_size}
  grayscale: {self.convert_to_grayscale}
  clahe: {self.apply_clahe}
  outliers_removed: {self.remove_outliers}
  
# Statistics
total_images_processed: {self.stats['total_processed']}
outliers_removed: {self.stats['outliers_removed']}
"""
        
        yaml_file = self.output_dir / 'data.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n📝 Configuration saved to {yaml_file}")
    
    def run(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🏥 CHEST X-RAY DATASET PREPROCESSING")
        print("="*70)
        
        # Load outlier list if needed
        if self.remove_outliers:
            self.load_outlier_list()
        
        # Process each split
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        # Generate comparison samples
        self.generate_comparison_samples(num_samples=5)
        
        # Save statistics
        self.save_statistics()
        
        # Create config
        self.create_yaml_config()
        
        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess and standardize chest X-ray dataset'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset directory (e.g., chest_xray_dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for cleaned dataset (e.g., chest_xray_clean)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target image size (default: 224x224)'
    )
    
    parser.add_argument(
        '--no-clahe',
        action='store_true',
        help='Disable CLAHE normalization'
    )
    
    parser.add_argument(
        '--keep-rgb',
        action='store_true',
        help='Keep RGB channels instead of converting to grayscale'
    )
    
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove outlier images (requires outlier CSV files)'
    )
    
    parser.add_argument(
        '--outlier-dir',
        type=str,
        default='analysis_results_sample',
        help='Directory containing outlier CSV files (default: analysis_results_sample)'
    )
    
    args = parser.parse_args()
    
    # Determine outlier list paths
    outlier_list = None
    if args.remove_outliers:
        outlier_dir = Path(args.outlier_dir)
        # We'll load all three outlier files
        outlier_list = outlier_dir / 'outlier_summary.csv'
    
    # Create preprocessor
    preprocessor = DatasetPreprocessor(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.size,
        apply_clahe=not args.no_clahe,
        convert_to_grayscale=not args.keep_rgb,
        remove_outliers=args.remove_outliers,
        outlier_list=str(outlier_list) if outlier_list else None
    )
    
    # Load outliers from all class files if needed
    if args.remove_outliers:
        outlier_dir = Path(args.outlier_dir)
        for class_name in ['normal', 'pneumonia', 'tuberculosis']:
            outlier_file = outlier_dir / f'outliers_{class_name}.csv'
            if outlier_file.exists():
                try:
                    df = pd.read_csv(outlier_file)
                    severe = df[df['outlier_count'] >= 3]
                    for _, row in severe.iterrows():
                        preprocessor.outliers_to_remove.add(row['filename'])
                except Exception as e:
                    print(f"Warning: Could not load {outlier_file}: {e}")
    
    # Run preprocessing
    preprocessor.run()


if __name__ == '__main__':
    main()
