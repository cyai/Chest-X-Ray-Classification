#!/usr/bin/env python3
"""
Chest X-Ray Dataset Standardization and Cleaning Script

This script:
1. Standardizes all images to consistent dimensions (224x224)
2. Converts all images to RGB (consistent color channels)
3. Applies CLAHE histogram equalization for brightness normalization
4. Removes detected outliers
5. Generates before/after comparison samples
6. Creates a cleaned dataset ready for retraining

Usage:
    python standardize_dataset.py --input chest_xray_dataset --output chest_xray_standardized
    python standardize_dataset.py --input chest_xray_dataset --output chest_xray_standardized --remove-outliers
"""

import os
import cv2
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class DatasetStandardizer:
    """Standardize and clean chest X-ray dataset"""
    
    def __init__(self, input_dir, output_dir, target_size=(224, 224), 
                 apply_clahe=True, remove_outliers=False, outlier_csv_dir=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.remove_outliers = remove_outliers
        self.outlier_csv_dir = Path(outlier_csv_dir) if outlier_csv_dir else None
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'skipped': 0,
            'errors': [],
            'size_changes': [],
            'brightness_changes': [],
            'removed_outliers': 0
        }
        
        # CLAHE settings
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Load outliers if needed
        self.outlier_files = set()
        if self.remove_outliers and self.outlier_csv_dir:
            self._load_outliers()
    
    def _load_outliers(self):
        """Load outlier filenames from CSV files"""
        print(f"Loading outlier files from {self.outlier_csv_dir}...")
        
        import pandas as pd
        
        for csv_file in self.outlier_csv_dir.glob('outliers_*.csv'):
            try:
                df = pd.read_csv(csv_file)
                # Get filenames with 3+ outlier metrics
                severe_outliers = df[df['outlier_count'] >= 3]['filename'].tolist()
                self.outlier_files.update(severe_outliers)
                print(f"  Loaded {len(severe_outliers)} outliers from {csv_file.name}")
            except Exception as e:
                print(f"  Warning: Could not load {csv_file.name}: {e}")
        
        print(f"Total outlier files to remove: {len(self.outlier_files)}")
    
    def standardize_image(self, image_path):
        """
        Standardize a single image
        
        Returns:
            standardized_image: Processed image
            before_stats: Statistics before processing
            after_stats: Statistics after processing
        """
        # Load image
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Record original stats
            before_stats = {
                'shape': img.shape,
                'mean': img.mean(),
                'std': img.std(),
                'dtype': str(img.dtype)
            }
            
            # Step 1: Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Step 2: Resize to target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Step 3: Apply CLAHE for brightness normalization
            if self.apply_clahe:
                # Convert to LAB color space
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel
                l, a, b = cv2.split(lab)
                l = self.clahe.apply(l)
                
                # Merge channels
                lab = cv2.merge([l, a, b])
                
                # Convert back to RGB
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Step 4: Normalize to 0-255 range
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Record processed stats
            after_stats = {
                'shape': img.shape,
                'mean': img.mean(),
                'std': img.std(),
                'dtype': str(img.dtype)
            }
            
            return img, before_stats, after_stats
            
        except Exception as e:
            raise RuntimeError(f"Error processing {image_path}: {str(e)}")
    
    def process_dataset(self, generate_samples=True, num_samples=10):
        """Process entire dataset"""
        
        print(f"{'='*70}")
        print("DATASET STANDARDIZATION")
        print(f"{'='*70}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target size: {self.target_size}")
        print(f"Apply CLAHE: {self.apply_clahe}")
        print(f"Remove outliers: {self.remove_outliers}")
        print(f"{'='*70}\n")
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['normal', 'pneumonia', 'tuberculosis']:
                output_path = self.output_dir / split / class_name
                output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        sample_images = defaultdict(list)
        
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*70}")
            print(f"Processing {split.upper()} set")
            print(f"{'='*70}")
            
            for class_name in ['normal', 'pneumonia', 'tuberculosis']:
                input_class_dir = self.input_dir / split / class_name
                output_class_dir = self.output_dir / split / class_name
                
                if not input_class_dir.exists():
                    print(f"  Warning: {input_class_dir} does not exist, skipping...")
                    continue
                
                # Get all image files
                image_files = list(input_class_dir.glob('*.jpg')) + \
                             list(input_class_dir.glob('*.jpeg')) + \
                             list(input_class_dir.glob('*.png'))
                
                print(f"\n  Class: {class_name}")
                print(f"  Found {len(image_files)} images")
                
                processed = 0
                skipped = 0
                
                for img_path in tqdm(image_files, desc=f"  Processing {class_name}"):
                    # Check if this is an outlier to remove
                    if self.remove_outliers and img_path.name in self.outlier_files:
                        skipped += 1
                        self.stats['removed_outliers'] += 1
                        continue
                    
                    try:
                        # Standardize image
                        std_img, before_stats, after_stats = self.standardize_image(img_path)
                        
                        # Save standardized image
                        output_path = output_class_dir / img_path.name
                        cv2.imwrite(str(output_path), cv2.cvtColor(std_img, cv2.COLOR_RGB2BGR))
                        
                        # Track statistics
                        self.stats['total_processed'] += 1
                        self.stats['size_changes'].append({
                            'before': before_stats['shape'],
                            'after': after_stats['shape']
                        })
                        self.stats['brightness_changes'].append({
                            'before': before_stats['mean'],
                            'after': after_stats['mean']
                        })
                        
                        # Collect samples for visualization
                        if generate_samples and len(sample_images[class_name]) < num_samples:
                            # Load original for comparison
                            original = cv2.imread(str(img_path))
                            if original is not None:
                                if len(original.shape) == 2:
                                    original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
                                else:
                                    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                                
                                sample_images[class_name].append({
                                    'filename': img_path.name,
                                    'original': original,
                                    'standardized': std_img,
                                    'before_stats': before_stats,
                                    'after_stats': after_stats
                                })
                        
                        processed += 1
                        
                    except Exception as e:
                        self.stats['errors'].append(f"{img_path}: {str(e)}")
                        skipped += 1
                
                print(f"  Processed: {processed}")
                print(f"  Skipped: {skipped}")
        
        # Generate comparison visualizations
        if generate_samples:
            self._generate_comparison_plots(sample_images)
        
        # Save statistics report
        self._save_statistics_report()
        
        print(f"\n{'='*70}")
        print("STANDARDIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Total images skipped: {self.stats['skipped']}")
        print(f"Total outliers removed: {self.stats['removed_outliers']}")
        print(f"Total errors: {len(self.stats['errors'])}")
        print(f"{'='*70}\n")
    
    def _generate_comparison_plots(self, sample_images):
        """Generate before/after comparison visualizations"""
        
        print(f"\n{'='*70}")
        print("Generating comparison visualizations...")
        print(f"{'='*70}")
        
        # Create comparison directory
        comparison_dir = self.output_dir / 'comparison_samples'
        comparison_dir.mkdir(exist_ok=True)
        
        for class_name, samples in sample_images.items():
            if not samples:
                continue
            
            # Create figure with before/after comparisons
            num_samples = len(samples)
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for idx, sample in enumerate(samples):
                # Original image
                axes[idx, 0].imshow(sample['original'])
                axes[idx, 0].set_title(
                    f"Original: {sample['filename']}\n"
                    f"Size: {sample['before_stats']['shape'][:2]}\n"
                    f"Mean: {sample['before_stats']['mean']:.1f}",
                    fontsize=10
                )
                axes[idx, 0].axis('off')
                
                # Standardized image
                axes[idx, 1].imshow(sample['standardized'])
                axes[idx, 1].set_title(
                    f"Standardized: {sample['filename']}\n"
                    f"Size: {sample['after_stats']['shape'][:2]}\n"
                    f"Mean: {sample['after_stats']['mean']:.1f}",
                    fontsize=10
                )
                axes[idx, 1].axis('off')
            
            plt.suptitle(f'{class_name.upper()} - Before/After Comparison', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(comparison_dir / f'{class_name}_comparison.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved comparison for {class_name}")
        
        # Generate statistics plots
        self._generate_statistics_plots(comparison_dir)
    
    def _generate_statistics_plots(self, output_dir):
        """Generate statistical analysis plots"""
        
        # Extract brightness changes
        before_brightness = [x['before'] for x in self.stats['brightness_changes']]
        after_brightness = [x['after'] for x in self.stats['brightness_changes']]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Brightness distribution comparison
        axes[0].hist(before_brightness, bins=50, alpha=0.6, label='Before', color='red')
        axes[0].hist(after_brightness, bins=50, alpha=0.6, label='After', color='green')
        axes[0].set_xlabel('Mean Brightness', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Brightness Distribution: Before vs After CLAHE', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Brightness change scatter
        axes[1].scatter(before_brightness, after_brightness, alpha=0.3, s=10)
        axes[1].plot([0, 255], [0, 255], 'r--', label='No change')
        axes[1].set_xlabel('Before CLAHE', fontsize=12)
        axes[1].set_ylabel('After CLAHE', fontsize=12)
        axes[1].set_title('Brightness Normalization Effect', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'brightness_normalization_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved brightness analysis")
    
    def _save_statistics_report(self):
        """Save detailed statistics report"""
        
        report_path = self.output_dir / 'standardization_report.json'
        
        # Calculate summary statistics
        before_brightness = [x['before'] for x in self.stats['brightness_changes']]
        after_brightness = [x['after'] for x in self.stats['brightness_changes']]
        
        report = {
            'total_processed': self.stats['total_processed'],
            'total_skipped': self.stats['skipped'],
            'outliers_removed': self.stats['removed_outliers'],
            'errors': len(self.stats['errors']),
            'target_size': self.target_size,
            'clahe_applied': self.apply_clahe,
            'brightness_stats': {
                'before': {
                    'mean': float(np.mean(before_brightness)),
                    'std': float(np.std(before_brightness)),
                    'min': float(np.min(before_brightness)),
                    'max': float(np.max(before_brightness))
                },
                'after': {
                    'mean': float(np.mean(after_brightness)),
                    'std': float(np.std(after_brightness)),
                    'min': float(np.min(after_brightness)),
                    'max': float(np.max(after_brightness))
                }
            },
            'error_details': self.stats['errors'][:100]  # First 100 errors
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  Saved statistics report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Standardize chest X-ray dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '--data-dir', 
                       default='chest_xray_dataset',
                       help='Input dataset directory')
    parser.add_argument('--output', '--output-dir',
                       default='chest_xray_standardized',
                       help='Output directory for standardized dataset')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (width height)')
    parser.add_argument('--no-clahe', action='store_true',
                       help='Disable CLAHE histogram equalization')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove detected outliers (requires outlier CSV files)')
    parser.add_argument('--outlier-dir', default='analysis_results_sample',
                       help='Directory containing outlier CSV files')
    parser.add_argument('--no-samples', action='store_true',
                       help='Skip generating comparison samples')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of comparison samples per class')
    
    args = parser.parse_args()
    
    # Create standardizer
    standardizer = DatasetStandardizer(
        input_dir=args.input,
        output_dir=args.output,
        target_size=tuple(args.size),
        apply_clahe=not args.no_clahe,
        remove_outliers=args.remove_outliers,
        outlier_csv_dir=args.outlier_dir if args.remove_outliers else None
    )
    
    # Process dataset
    standardizer.process_dataset(
        generate_samples=not args.no_samples,
        num_samples=args.num_samples
    )
    
    print("\n✅ Dataset standardization complete!")
    print(f"\nStandardized dataset saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review comparison samples in: chest_xray_standardized/comparison_samples/")
    print("  2. Check standardization report: chest_xray_standardized/standardization_report.json")
    print("  3. Train models using the standardized dataset")


if __name__ == '__main__':
    main()
