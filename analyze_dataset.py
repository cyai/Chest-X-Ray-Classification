#!/usr/bin/env python3
"""
Comprehensive Chest X-Ray Dataset Analysis Script
Detects irregularities, anomalies, and distribution issues in medical imaging datasets

Usage:
    python analyze_dataset.py --data-dir chest_xray_dataset --output-dir analysis_results
    python analyze_dataset.py --data-dir chest_xray_dataset --sample-size 1000 --z-threshold 3.0
    python analyze_dataset.py --data-dir chest_xray_dataset --full-analysis --generate-plots
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DatasetAnalyzer:
    """Comprehensive medical image dataset analyzer"""
    
    def __init__(self, data_dir: str, output_dir: str = "analysis_results",
                 z_threshold: float = 3.0, sample_size: int = None):
        """
        Initialize dataset analyzer
        
        Args:
            data_dir: Path to dataset directory (expects train/val/test subdirs)
            output_dir: Directory to save analysis results
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
            sample_size: Number of images to sample per class (None = all images)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.z_threshold = z_threshold
        self.sample_size = sample_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for analysis results
        self.image_data = []
        self.class_stats = {}
        self.outliers = defaultdict(list)
        self.distribution_analysis = {}
        
        # Class names
        self.classes = ['normal', 'pneumonia', 'tuberculosis']
        self.splits = ['train', 'val', 'test']
        
        print(f"📊 Dataset Analyzer initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Z-score threshold: {self.z_threshold}")
        print(f"   Sample size per class: {self.sample_size or 'All images'}")
    
    def collect_image_paths(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Collect all image paths organized by split and class
        
        Returns:
            Nested dict: {split: {class: [path1, path2, ...]}}
        """
        image_paths = {split: {cls: [] for cls in self.classes} for split in self.splits}
        
        print("\n🔍 Collecting image paths...")
        for split in self.splits:
            for cls in self.classes:
                class_dir = self.data_dir / split / cls
                if not class_dir.exists():
                    print(f"⚠️  Warning: Directory not found: {class_dir}")
                    continue
                
                # Get all image files
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                paths = []
                for ext in extensions:
                    paths.extend(list(class_dir.glob(ext)))
                
                # Sample if requested
                if self.sample_size and len(paths) > self.sample_size:
                    paths = np.random.choice(paths, self.sample_size, replace=False).tolist()
                
                image_paths[split][cls] = paths
                print(f"   {split}/{cls}: {len(paths)} images")
        
        return image_paths
    
    def analyze_image(self, img_path: Path, split: str, cls: str) -> Dict[str, Any]:
        """
        Analyze individual image properties
        
        Args:
            img_path: Path to image file
            split: Dataset split (train/val/test)
            cls: Class name
            
        Returns:
            Dictionary of image properties
        """
        try:
            # Load image with PIL
            img_pil = Image.open(img_path)
            img_array = np.array(img_pil)
            
            # Load with OpenCV for advanced analysis
            img_cv = cv2.imread(str(img_path))
            
            # Basic properties
            height, width = img_array.shape[:2]
            channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
            file_size = img_path.stat().st_size / 1024  # KB
            
            # Color mode
            mode = img_pil.mode
            
            # Convert to RGB if needed for analysis
            if len(img_array.shape) == 2:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img_array
            
            # Color statistics
            mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))
            std_r, std_g, std_b = img_rgb.std(axis=(0, 1))
            
            # Grayscale conversion for intensity analysis
            if img_cv is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Intensity statistics
            mean_intensity = gray.mean()
            std_intensity = gray.std()
            min_intensity = gray.min()
            max_intensity = gray.max()
            
            # Brightness and contrast
            brightness = mean_intensity
            contrast = std_intensity
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Edge density (Canny edge detection)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = (edges > 0).sum() / edges.size
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            hist_entropy = stats.entropy(hist + 1e-10)
            
            # Dynamic range
            dynamic_range = max_intensity - min_intensity
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Check for uniformity (potential corruption)
            unique_values = len(np.unique(gray))
            is_uniform = unique_values < 10  # Flag if very few unique values
            
            # Color channel correlation (check for grayscale vs RGB)
            channel_variance = np.var([mean_r, mean_g, mean_b])
            is_grayscale_like = channel_variance < 10  # Channels very similar
            
            return {
                'path': str(img_path),
                'filename': img_path.name,
                'split': split,
                'class': cls,
                # Dimensions
                'width': width,
                'height': height,
                'channels': channels,
                'aspect_ratio': aspect_ratio,
                'total_pixels': width * height,
                # File properties
                'file_size_kb': file_size,
                'mode': mode,
                # Color statistics
                'mean_r': mean_r,
                'mean_g': mean_g,
                'mean_b': mean_b,
                'std_r': std_r,
                'std_g': std_g,
                'std_b': std_b,
                'channel_variance': channel_variance,
                'is_grayscale_like': is_grayscale_like,
                # Intensity statistics
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'dynamic_range': dynamic_range,
                # Quality metrics
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'edge_density': edge_density,
                'hist_entropy': hist_entropy,
                # Anomaly flags
                'unique_values': unique_values,
                'is_uniform': is_uniform,
                'is_valid': True
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {img_path}: {e}")
            return {
                'path': str(img_path),
                'filename': img_path.name,
                'split': split,
                'class': cls,
                'is_valid': False,
                'error': str(e)
            }
    
    def analyze_all_images(self, image_paths: Dict[str, Dict[str, List[Path]]]) -> pd.DataFrame:
        """
        Analyze all images in the dataset
        
        Args:
            image_paths: Dictionary of image paths by split and class
            
        Returns:
            DataFrame with all image properties
        """
        print("\n🔬 Analyzing image properties...")
        
        all_data = []
        total_images = sum(len(paths) for split_data in image_paths.values() 
                          for paths in split_data.values())
        
        with tqdm(total=total_images, desc="Processing images") as pbar:
            for split in self.splits:
                for cls in self.classes:
                    for img_path in image_paths[split][cls]:
                        img_data = self.analyze_image(img_path, split, cls)
                        all_data.append(img_data)
                        pbar.update(1)
        
        df = pd.DataFrame(all_data)
        
        # Filter valid images
        valid_df = df[df['is_valid'] == True].copy()
        invalid_count = len(df) - len(valid_df)
        
        if invalid_count > 0:
            print(f"⚠️  Found {invalid_count} invalid/corrupted images")
            invalid_df = df[df['is_valid'] == False]
            invalid_df[['path', 'error']].to_csv(
                self.output_dir / 'invalid_images.csv', index=False
            )
        
        print(f"✅ Successfully analyzed {len(valid_df)} images")
        
        return valid_df
    
    def compute_class_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistical summaries for each class
        
        Args:
            df: DataFrame with image properties
            
        Returns:
            Dictionary of statistics per class
        """
        print("\n📈 Computing class statistics...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove binary flags from statistics
        numeric_cols = [col for col in numeric_cols if col not in 
                       ['is_valid', 'is_uniform', 'is_grayscale_like', 'channels']]
        
        stats_dict = {}
        
        for cls in self.classes:
            class_df = df[df['class'] == cls]
            
            if len(class_df) == 0:
                continue
            
            stats_dict[cls] = {
                'count': len(class_df),
                'mean': class_df[numeric_cols].mean().to_dict(),
                'std': class_df[numeric_cols].std().to_dict(),
                'min': class_df[numeric_cols].min().to_dict(),
                'max': class_df[numeric_cols].max().to_dict(),
                'median': class_df[numeric_cols].median().to_dict(),
                'q25': class_df[numeric_cols].quantile(0.25).to_dict(),
                'q75': class_df[numeric_cols].quantile(0.75).to_dict(),
            }
            
            # Additional flags - convert to native Python int
            stats_dict[cls]['uniform_images'] = int(class_df['is_uniform'].sum())
            stats_dict[cls]['grayscale_like_images'] = int(class_df['is_grayscale_like'].sum())
        
        # Convert all numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        stats_dict = convert_to_native(stats_dict)
        
        # Save to JSON
        with open(self.output_dir / 'class_statistics.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print("✅ Class statistics computed and saved")
        
        return stats_dict
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers using z-score method
        
        Args:
            df: DataFrame with image properties
            
        Returns:
            Dictionary of outlier DataFrames per class
        """
        print(f"\n🎯 Detecting outliers (z-score > {self.z_threshold})...")
        
        numeric_cols = ['width', 'height', 'aspect_ratio', 'file_size_kb',
                       'mean_intensity', 'std_intensity', 'dynamic_range',
                       'brightness', 'contrast', 'sharpness', 'edge_density',
                       'mean_r', 'mean_g', 'mean_b', 'channel_variance']
        
        outlier_dfs = {}
        outlier_summary = []
        
        for cls in self.classes:
            class_df = df[df['class'] == cls].copy()
            
            if len(class_df) == 0:
                continue
            
            # Compute z-scores for each metric
            outlier_flags = pd.DataFrame(index=class_df.index)
            
            for col in numeric_cols:
                if col in class_df.columns:
                    z_scores = np.abs(stats.zscore(class_df[col], nan_policy='omit'))
                    outlier_flags[f'{col}_outlier'] = z_scores > self.z_threshold
                    outlier_flags[f'{col}_zscore'] = z_scores
            
            # Count outliers per image
            outlier_cols = [col for col in outlier_flags.columns if col.endswith('_outlier')]
            class_df['outlier_count'] = outlier_flags[outlier_cols].sum(axis=1)
            class_df['is_outlier'] = class_df['outlier_count'] > 0
            
            # Add z-scores
            zscore_cols = [col for col in outlier_flags.columns if col.endswith('_zscore')]
            for col in zscore_cols:
                class_df[col] = outlier_flags[col]
            
            # Get outliers
            outliers = class_df[class_df['is_outlier']].copy()
            outliers = outliers.sort_values('outlier_count', ascending=False)
            
            outlier_dfs[cls] = outliers
            
            print(f"   {cls}: {len(outliers)} outliers ({len(outliers)/len(class_df)*100:.2f}%)")
            
            # Summary
            outlier_summary.append({
                'class': cls,
                'total_images': len(class_df),
                'outlier_images': len(outliers),
                'outlier_percentage': len(outliers) / len(class_df) * 100,
                'severe_outliers': len(outliers[outliers['outlier_count'] >= 3]),
            })
        
        # Save outlier summary
        summary_df = pd.DataFrame(outlier_summary)
        summary_df.to_csv(self.output_dir / 'outlier_summary.csv', index=False)
        
        # Save detailed outlier lists
        for cls, outliers in outlier_dfs.items():
            if len(outliers) > 0:
                outliers.to_csv(
                    self.output_dir / f'outliers_{cls}.csv', index=False
                )
        
        print("✅ Outlier detection complete")
        
        return outlier_dfs
    
    def analyze_distribution_mismatch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze class distribution across train/val/test splits
        
        Args:
            df: DataFrame with image properties
            
        Returns:
            Dictionary with distribution analysis
        """
        print("\n⚖️  Analyzing class distribution across splits...")
        
        distribution = {}
        
        for split in self.splits:
            split_df = df[df['split'] == split]
            total = len(split_df)
            
            distribution[split] = {
                'total': total,
                'classes': {}
            }
            
            for cls in self.classes:
                count = len(split_df[split_df['class'] == cls])
                percentage = (count / total * 100) if total > 0 else 0
                
                distribution[split]['classes'][cls] = {
                    'count': count,
                    'percentage': percentage
                }
        
        # Compute distribution mismatch
        train_dist = {cls: distribution['train']['classes'][cls]['percentage'] 
                     for cls in self.classes}
        test_dist = {cls: distribution['test']['classes'][cls]['percentage'] 
                    for cls in self.classes}
        
        mismatch = {cls: abs(train_dist[cls] - test_dist[cls]) 
                   for cls in self.classes}
        
        distribution['mismatch'] = {
            'train_distribution': train_dist,
            'test_distribution': test_dist,
            'absolute_difference': mismatch,
            'max_mismatch': max(mismatch.values()),
            'problematic_class': max(mismatch, key=mismatch.get)
        }
        
        # Save distribution analysis
        with open(self.output_dir / 'distribution_analysis.json', 'w') as f:
            json.dump(distribution, f, indent=2)
        
        print(f"   Max distribution mismatch: {distribution['mismatch']['max_mismatch']:.2f}%")
        print(f"   Problematic class: {distribution['mismatch']['problematic_class']}")
        
        return distribution
    
    def cross_class_comparison(self, df: pd.DataFrame, class_stats: Dict) -> Dict[str, Any]:
        """
        Compare classes to identify systematic differences
        
        Args:
            df: DataFrame with image properties
            class_stats: Dictionary of class statistics
            
        Returns:
            Dictionary with comparison results
        """
        print("\n🔬 Performing cross-class comparison...")
        
        comparison = {}
        
        # Metrics to compare
        metrics = ['width', 'height', 'aspect_ratio', 'file_size_kb',
                  'mean_intensity', 'brightness', 'contrast', 'sharpness',
                  'edge_density', 'mean_r', 'mean_g', 'mean_b', 'channel_variance']
        
        for metric in metrics:
            metric_comparison = {}
            
            for cls in self.classes:
                if cls in class_stats:
                    metric_comparison[cls] = {
                        'mean': class_stats[cls]['mean'].get(metric, 0),
                        'std': class_stats[cls]['std'].get(metric, 0),
                        'min': class_stats[cls]['min'].get(metric, 0),
                        'max': class_stats[cls]['max'].get(metric, 0),
                    }
            
            # Compute coefficient of variation across classes
            means = [metric_comparison[cls]['mean'] for cls in self.classes 
                    if cls in metric_comparison]
            
            if len(means) > 0 and np.mean(means) > 0:
                cv = np.std(means) / np.mean(means) * 100
                metric_comparison['coefficient_of_variation'] = cv
                metric_comparison['is_significantly_different'] = cv > 20  # >20% variation
            
            comparison[metric] = metric_comparison
        
        # Statistical tests (ANOVA)
        print("\n   Running ANOVA tests for significant differences...")
        anova_results = {}
        
        for metric in metrics:
            groups = [df[df['class'] == cls][metric].dropna() 
                     for cls in self.classes]
            
            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[metric] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05
                    }
                except Exception as e:
                    anova_results[metric] = {'error': str(e)}
        
        comparison['anova_tests'] = anova_results
        
        # Save comparison results
        with open(self.output_dir / 'cross_class_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Count significant differences
        sig_count = sum(1 for result in anova_results.values() 
                       if isinstance(result, dict) and result.get('is_significant', False))
        
        print(f"   Found {sig_count}/{len(metrics)} metrics with significant class differences")
        
        return comparison
    
    def generate_visualizations(self, df: pd.DataFrame, outlier_dfs: Dict):
        """
        Generate comprehensive visualizations
        
        Args:
            df: DataFrame with image properties
            outlier_dfs: Dictionary of outlier DataFrames
        """
        print("\n📊 Generating visualizations...")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Class distribution across splits
        self._plot_class_distribution(df, viz_dir)
        
        # 2. Box plots for key metrics
        self._plot_metric_comparison(df, viz_dir)
        
        # 3. Correlation heatmap per class
        self._plot_correlation_heatmaps(df, viz_dir)
        
        # 4. Outlier distribution
        self._plot_outlier_distribution(outlier_dfs, viz_dir)
        
        # 5. RGB distribution comparison
        self._plot_rgb_distributions(df, viz_dir)
        
        # 6. Dimension scatter plots
        self._plot_dimension_analysis(df, viz_dir)
        
        # 7. Quality metrics comparison
        self._plot_quality_metrics(df, viz_dir)
        
        print(f"✅ Visualizations saved to {viz_dir}")
    
    def _plot_class_distribution(self, df: pd.DataFrame, viz_dir: Path):
        """Plot class distribution across splits"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        split_order = ['train', 'val', 'test']
        class_order = ['normal', 'pneumonia', 'tuberculosis']
        
        split_class_counts = df.groupby(['split', 'class']).size().reset_index(name='count')
        
        ax = axes[0]
        for cls in class_order:
            cls_data = split_class_counts[split_class_counts['class'] == cls]
            ax.bar(cls_data['split'], cls_data['count'], label=cls.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Dataset Split')
        ax.set_ylabel('Number of Images')
        ax.set_title('Class Distribution Across Splits (Absolute Counts)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Percentage plot
        ax = axes[1]
        split_totals = df.groupby('split').size()
        
        for cls in class_order:
            percentages = []
            for split in split_order:
                count = len(df[(df['split'] == split) & (df['class'] == cls)])
                total = split_totals[split]
                percentages.append(count / total * 100 if total > 0 else 0)
            ax.plot(split_order, percentages, marker='o', label=cls.capitalize(), linewidth=2)
        
        ax.set_xlabel('Dataset Split')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Class Distribution Across Splits (Percentages)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_comparison(self, df: pd.DataFrame, viz_dir: Path):
        """Plot box plots comparing key metrics across classes"""
        metrics = [
            ('brightness', 'Brightness'),
            ('contrast', 'Contrast'),
            ('sharpness', 'Sharpness'),
            ('aspect_ratio', 'Aspect Ratio'),
            ('edge_density', 'Edge Density'),
            ('channel_variance', 'Channel Variance')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            data_to_plot = [df[df['class'] == cls][metric].dropna() 
                           for cls in self.classes]
            
            bp = ax.boxplot(data_to_plot, labels=[c.capitalize() for c in self.classes],
                           patch_artist=True, showfliers=True)
            
            # Color boxes
            colors = ['#3498db', '#e74c3c', '#f39c12']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel(title)
            ax.set_title(f'{title} Distribution by Class')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'metric_comparison_boxplots.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, df: pd.DataFrame, viz_dir: Path):
        """Plot correlation heatmaps for each class"""
        metrics = ['width', 'height', 'brightness', 'contrast', 'sharpness',
                  'edge_density', 'mean_intensity', 'std_intensity']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, cls in enumerate(self.classes):
            class_df = df[df['class'] == cls][metrics]
            corr = class_df.corr()
            
            ax = axes[idx]
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'{cls.capitalize()} - Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_heatmaps.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_outlier_distribution(self, outlier_dfs: Dict, viz_dir: Path):
        """Plot outlier distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        outlier_counts = {cls: len(df) for cls, df in outlier_dfs.items()}
        
        classes = list(outlier_counts.keys())
        counts = list(outlier_counts.values())
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        bars = ax.bar([c.capitalize() for c in classes], counts, color=colors, alpha=0.7)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Outlier Images')
        ax.set_title(f'Outlier Detection Results (Z-score > {self.z_threshold})')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'outlier_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rgb_distributions(self, df: pd.DataFrame, viz_dir: Path):
        """Plot RGB channel distributions"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        channels = [('mean_r', 'Red'), ('mean_g', 'Green'), ('mean_b', 'Blue')]
        colors = ['red', 'green', 'blue']
        
        for idx, ((channel, label), color) in enumerate(zip(channels, colors)):
            ax = axes[idx]
            
            for cls in self.classes:
                class_data = df[df['class'] == cls][channel].dropna()
                ax.hist(class_data, bins=50, alpha=0.5, label=cls.capitalize(),
                       density=True)
            
            ax.set_xlabel(f'{label} Channel Mean Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{label} Channel Distribution by Class')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'rgb_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_dimension_analysis(self, df: pd.DataFrame, viz_dir: Path):
        """Plot dimension scatter plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Width vs Height scatter
        ax = axes[0]
        for cls in self.classes:
            class_df = df[df['class'] == cls]
            ax.scatter(class_df['width'], class_df['height'], 
                      alpha=0.4, s=20, label=cls.capitalize())
        
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title('Image Dimensions Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Aspect ratio histogram
        ax = axes[1]
        for cls in self.classes:
            class_data = df[df['class'] == cls]['aspect_ratio'].dropna()
            ax.hist(class_data, bins=50, alpha=0.5, label=cls.capitalize())
        
        ax.set_xlabel('Aspect Ratio (width/height)')
        ax.set_ylabel('Frequency')
        ax.set_title('Aspect Ratio Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Square (1:1)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'dimension_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, df: pd.DataFrame, viz_dir: Path):
        """Plot quality metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sharpness vs Contrast
        ax = axes[0, 0]
        for cls in self.classes:
            class_df = df[df['class'] == cls]
            ax.scatter(class_df['contrast'], class_df['sharpness'],
                      alpha=0.4, s=20, label=cls.capitalize())
        ax.set_xlabel('Contrast')
        ax.set_ylabel('Sharpness')
        ax.set_title('Sharpness vs Contrast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Brightness distribution
        ax = axes[0, 1]
        for cls in self.classes:
            class_data = df[df['class'] == cls]['brightness'].dropna()
            ax.hist(class_data, bins=50, alpha=0.5, label=cls.capitalize())
        ax.set_xlabel('Brightness')
        ax.set_ylabel('Frequency')
        ax.set_title('Brightness Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Edge density
        ax = axes[1, 0]
        for cls in self.classes:
            class_data = df[df['class'] == cls]['edge_density'].dropna()
            ax.hist(class_data, bins=50, alpha=0.5, label=cls.capitalize())
        ax.set_xlabel('Edge Density')
        ax.set_ylabel('Frequency')
        ax.set_title('Edge Density Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Dynamic range
        ax = axes[1, 1]
        data_to_plot = [df[df['class'] == cls]['dynamic_range'].dropna() 
                       for cls in self.classes]
        bp = ax.boxplot(data_to_plot, labels=[c.capitalize() for c in self.classes],
                       patch_artist=True)
        colors = ['#3498db', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Dynamic Range')
        ax.set_title('Dynamic Range by Class')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'quality_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_remediation_report(self, df: pd.DataFrame, outlier_dfs: Dict,
                                   distribution: Dict, comparison: Dict) -> str:
        """
        Generate comprehensive remediation recommendations
        
        Args:
            df: DataFrame with image properties
            outlier_dfs: Dictionary of outlier DataFrames
            distribution: Distribution analysis results
            comparison: Cross-class comparison results
            
        Returns:
            Markdown-formatted report
        """
        print("\n📝 Generating remediation report...")
        
        report = []
        report.append("# Dataset Analysis & Remediation Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Dataset:** {self.data_dir}")
        report.append(f"\n**Total Images Analyzed:** {len(df)}")
        report.append("\n---\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        total_outliers = sum(len(df) for df in outlier_dfs.values())
        outlier_percentage = total_outliers / len(df) * 100
        
        report.append(f"- **Total Outliers Detected:** {total_outliers} ({outlier_percentage:.2f}%)")
        report.append(f"- **Z-score Threshold:** {self.z_threshold}")
        report.append(f"- **Max Distribution Mismatch:** {distribution['mismatch']['max_mismatch']:.2f}% ({distribution['mismatch']['problematic_class']})")
        
        # Count significant differences
        sig_diffs = sum(1 for result in comparison.get('anova_tests', {}).values()
                       if isinstance(result, dict) and result.get('is_significant', False))
        report.append(f"- **Significant Cross-Class Differences:** {sig_diffs} metrics")
        report.append("\n")
        
        # Critical Issues
        report.append("## 🚨 Critical Issues Identified\n")
        
        issues = []
        
        # Issue 1: Distribution mismatch
        if distribution['mismatch']['max_mismatch'] > 10:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Train/Test Distribution Mismatch',
                'details': f"{distribution['mismatch']['problematic_class'].capitalize()} class has {distribution['mismatch']['max_mismatch']:.1f}% distribution difference between train and test sets",
                'impact': 'Models learn biased class priors, leading to poor generalization'
            })
        
        # Issue 2: Class-specific outliers
        for cls, outliers in outlier_dfs.items():
            severe_outliers = len(outliers[outliers['outlier_count'] >= 3])
            if severe_outliers > 0:
                issues.append({
                    'severity': 'MEDIUM',
                    'issue': f'{cls.capitalize()} Class Quality Issues',
                    'details': f'{severe_outliers} images with 3+ anomalous properties',
                    'impact': 'Low-quality images may confuse training and reduce model accuracy'
                })
        
        # Issue 3: Significant cross-class differences
        sig_metrics = [metric for metric, result in comparison.get('anova_tests', {}).items()
                      if isinstance(result, dict) and result.get('is_significant')]
        
        if 'brightness' in sig_metrics or 'contrast' in sig_metrics:
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Systematic Brightness/Contrast Differences',
                'details': 'Classes show significantly different brightness/contrast profiles',
                'impact': 'Model may learn spurious brightness cues instead of disease patterns'
            })
        
        if 'channel_variance' in sig_metrics:
            issues.append({
                'severity': 'LOW',
                'issue': 'Color Channel Inconsistencies',
                'details': 'Some classes more grayscale-like than others',
                'impact': 'Indicates inconsistent preprocessing or source datasets'
            })
        
        for i, issue in enumerate(issues, 1):
            report.append(f"### {i}. [{issue['severity']}] {issue['issue']}\n")
            report.append(f"**Details:** {issue['details']}\n")
            report.append(f"**Impact:** {issue['impact']}\n")
        
        # Detailed Findings
        report.append("\n## 📊 Detailed Findings\n")
        
        # Distribution breakdown
        report.append("### Class Distribution Across Splits\n")
        report.append("| Split | Normal | Pneumonia | Tuberculosis | Total |")
        report.append("|-------|--------|-----------|--------------|-------|")
        
        for split in ['train', 'val', 'test']:
            split_data = distribution[split]['classes']
            total = distribution[split]['total']
            row = f"| {split.capitalize()} | "
            row += f"{split_data['normal']['count']} ({split_data['normal']['percentage']:.1f}%) | "
            row += f"{split_data['pneumonia']['count']} ({split_data['pneumonia']['percentage']:.1f}%) | "
            row += f"{split_data['tuberculosis']['count']} ({split_data['tuberculosis']['percentage']:.1f}%) | "
            row += f"{total} |"
            report.append(row)
        
        report.append("\n")
        
        # Outlier breakdown
        report.append("### Outlier Detection Results\n")
        report.append("| Class | Total Images | Outliers | Percentage | Severe (3+) |")
        report.append("|-------|--------------|----------|------------|-------------|")
        
        for cls in self.classes:
            total = len(df[df['class'] == cls])
            outliers = outlier_dfs.get(cls, pd.DataFrame())
            outlier_count = len(outliers)
            percentage = (outlier_count / total * 100) if total > 0 else 0
            severe = len(outliers[outliers['outlier_count'] >= 3]) if len(outliers) > 0 else 0
            
            report.append(f"| {cls.capitalize()} | {total} | {outlier_count} | {percentage:.2f}% | {severe} |")
        
        report.append("\n")
        
        # Recommendations
        report.append("## ✅ Remediation Recommendations\n")
        
        report.append("### Priority 1: Fix Distribution Mismatch (CRITICAL)\n")
        report.append("**Problem:** Training and test sets have significantly different class distributions.\n")
        report.append("**Solution Options:**\n")
        report.append("1. **Stratified Resampling:** Use weighted random sampling to match test distribution during training")
        report.append("   ```python")
        report.append("   from torch.utils.data import WeightedRandomSampler")
        report.append("   # Target distribution: 36% Normal, 23% Pneumonia, 41% TB")
        report.append("   class_weights = compute_weights_for_target_distribution()")
        report.append("   sampler = WeightedRandomSampler(class_weights, len(dataset))")
        report.append("   ```")
        report.append("2. **Weighted Loss Function:** Apply inverse class frequency weights")
        report.append("   ```python")
        report.append("   # Weight classes to match test importance")
        report.append("   class_weights = torch.tensor([1.0/0.36, 1.0/0.23, 1.0/0.41])")
        report.append("   criterion = nn.CrossEntropyLoss(weight=class_weights)")
        report.append("   ```")
        report.append("3. **Redistribute Dataset:** Reorganize train/val/test splits to have consistent distributions")
        report.append("\n**Expected Impact:** +5-10% accuracy improvement, significantly better TB recall\n")
        
        report.append("### Priority 2: Remove/Fix Outlier Images\n")
        report.append(f"**Problem:** {total_outliers} images ({outlier_percentage:.2f}%) have anomalous properties.\n")
        report.append("**Solution:**\n")
        report.append("1. **Review Severe Outliers:** Manually inspect images with 3+ anomalous metrics")
        report.append(f"   - Check files: `outliers_*.csv` in {self.output_dir}")
        report.append("2. **Automated Filtering:** Remove images with extreme anomalies")
        report.append("   ```python")
        report.append("   # Remove images that are uniform (corrupted)")
        report.append("   # Remove images with extreme brightness/contrast")
        report.append("   # Remove images with unusual aspect ratios")
        report.append("   ```")
        report.append("3. **Preprocessing Pipeline:** Standardize all images")
        report.append("   - Resize to consistent dimensions (e.g., 224x224)")
        report.append("   - Apply CLAHE (histogram equalization) for brightness normalization")
        report.append("   - Convert all to grayscale or ensure consistent RGB channels")
        report.append("\n**Expected Impact:** +2-4% accuracy improvement, more stable training\n")
        
        report.append("### Priority 3: Normalize Class-Specific Differences\n")
        report.append("**Problem:** Significant differences in brightness, contrast, and color between classes.\n")
        report.append("**Solution:**\n")
        report.append("1. **Advanced Preprocessing:**")
        report.append("   ```python")
        report.append("   from skimage import exposure")
        report.append("   # Apply CLAHE to normalize brightness/contrast")
        report.append("   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))")
        report.append("   normalized = clahe.apply(image)")
        report.append("   ```")
        report.append("2. **Color Standardization:** Convert all images to grayscale or ensure RGB consistency")
        report.append("3. **Intensity Normalization:** Z-score normalize pixel values per image")
        report.append("\n**Expected Impact:** Model learns anatomical features instead of brightness cues\n")
        
        report.append("### Priority 4: Data Augmentation Strategy\n")
        report.append("**Recommendation:** Focus augmentation on minority class (TB) and balance dataset\n")
        report.append("```python")
        report.append("# Class-specific augmentation")
        report.append("tb_transforms = transforms.Compose([")
        report.append("    transforms.RandomHorizontalFlip(p=0.5),")
        report.append("    transforms.RandomRotation(10),")
        report.append("    transforms.ColorJitter(brightness=0.2, contrast=0.2),")
        report.append("    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),")
        report.append("    # Apply 2x more augmentation to TB class")
        report.append("])")
        report.append("```\n")
        
        report.append("### Priority 5: Ensemble Strategy\n")
        report.append("**Recommendation:** Combine existing models to leverage their strengths\n")
        report.append("- Use RNN model for Pneumonia detection (97% recall)")
        report.append("- Use ViT model for Normal detection (88% recall)")
        report.append("- Use Active Learning model for TB detection")
        report.append("- Weighted voting based on per-class strengths")
        report.append("\n**Expected Impact:** +2-5% accuracy improvement without retraining\n")
        
        # Specific files to review
        report.append("## 📁 Files Requiring Manual Review\n")
        
        for cls, outliers in outlier_dfs.items():
            severe = outliers[outliers['outlier_count'] >= 5].sort_values('outlier_count', ascending=False)
            
            if len(severe) > 0:
                report.append(f"### {cls.capitalize()} Class - Top Problematic Images\n")
                report.append("| Filename | Outlier Count | Issues |")
                report.append("|----------|---------------|--------|")
                
                for _, row in severe.head(10).iterrows():
                    # Identify which metrics are outliers
                    issues = []
                    if row.get('brightness_zscore', 0) > self.z_threshold:
                        issues.append('brightness')
                    if row.get('contrast_zscore', 0) > self.z_threshold:
                        issues.append('contrast')
                    if row.get('sharpness_zscore', 0) > self.z_threshold:
                        issues.append('sharpness')
                    if row.get('aspect_ratio_zscore', 0) > self.z_threshold:
                        issues.append('dimensions')
                    
                    issues_str = ', '.join(issues[:3])
                    if len(issues) > 3:
                        issues_str += '...'
                    
                    report.append(f"| `{row['filename']}` | {int(row['outlier_count'])} | {issues_str} |")
                
                report.append("\n")
        
        # Implementation Roadmap
        report.append("## 🗺️ Implementation Roadmap\n")
        report.append("### Week 1: Quick Wins")
        report.append("- [ ] Implement weighted sampling for class balance")
        report.append("- [ ] Apply weighted loss function")
        report.append("- [ ] Retrain best model (ViT or RNN)")
        report.append("- [ ] **Expected:** 77-80% accuracy\n")
        
        report.append("### Week 2: Data Cleaning")
        report.append("- [ ] Remove severe outliers (manual review)")
        report.append("- [ ] Implement CLAHE preprocessing")
        report.append("- [ ] Standardize image dimensions")
        report.append("- [ ] Retrain all models")
        report.append("- [ ] **Expected:** 80-82% accuracy\n")
        
        report.append("### Week 3: Advanced Techniques")
        report.append("- [ ] Implement model ensemble")
        report.append("- [ ] Add focused TB augmentation")
        report.append("- [ ] Experiment with hybrid architectures")
        report.append("- [ ] **Expected:** 82-85% accuracy\n")
        
        report.append("### Week 4: Validation & Deployment")
        report.append("- [ ] Cross-validation on cleaned dataset")
        report.append("- [ ] Implement uncertainty quantification")
        report.append("- [ ] Prepare production pipeline")
        report.append("- [ ] **Target:** 85%+ accuracy, 70%+ TB recall\n")
        
        # Appendix
        report.append("## 📎 Appendix: Generated Files\n")
        report.append(f"All analysis results saved to: `{self.output_dir}/`\n")
        report.append("- `class_statistics.json` - Detailed statistics per class")
        report.append("- `distribution_analysis.json` - Split distribution analysis")
        report.append("- `cross_class_comparison.json` - Statistical comparisons")
        report.append("- `outlier_summary.csv` - Outlier count summary")
        report.append("- `outliers_*.csv` - Detailed outlier lists per class")
        report.append("- `visualizations/` - All generated plots")
        report.append("\n")
        
        report_text = '\n'.join(report)
        
        # Save report
        report_path = self.output_dir / 'remediation_report.md'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Remediation report saved to {report_path}")
        
        return report_text
    
    def run_full_analysis(self, generate_plots: bool = True):
        """
        Run complete dataset analysis pipeline
        
        Args:
            generate_plots: Whether to generate visualization plots
        """
        print("\n" + "="*70)
        print("🏥 CHEST X-RAY DATASET ANALYSIS")
        print("="*70)
        
        # Step 1: Collect image paths
        image_paths = self.collect_image_paths()
        
        # Step 2: Analyze all images
        df = self.analyze_all_images(image_paths)
        
        # Save raw data
        df.to_csv(self.output_dir / 'image_analysis_data.csv', index=False)
        print(f"\n💾 Raw analysis data saved to {self.output_dir / 'image_analysis_data.csv'}")
        
        # Step 3: Compute class statistics
        class_stats = self.compute_class_statistics(df)
        
        # Step 4: Detect outliers
        outlier_dfs = self.detect_outliers(df)
        
        # Step 5: Analyze distribution
        distribution = self.analyze_distribution_mismatch(df)
        
        # Step 6: Cross-class comparison
        comparison = self.cross_class_comparison(df, class_stats)
        
        # Step 7: Generate visualizations
        if generate_plots:
            self.generate_visualizations(df, outlier_dfs)
        
        # Step 8: Generate remediation report
        report = self.generate_remediation_report(df, outlier_dfs, distribution, comparison)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📁 All results saved to: {self.output_dir}")
        print(f"\n📋 Next steps:")
        print(f"   1. Review remediation report: {self.output_dir / 'remediation_report.md'}")
        print(f"   2. Check outlier files: {self.output_dir / 'outliers_*.csv'}")
        print(f"   3. View visualizations: {self.output_dir / 'visualizations/'}")
        print(f"   4. Implement recommended fixes")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chest X-ray dataset for irregularities and quality issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis with sampling
  python analyze_dataset.py --data-dir chest_xray_dataset --sample-size 1000

  # Full analysis with all images
  python analyze_dataset.py --data-dir chest_xray_dataset --full-analysis

  # Custom z-score threshold
  python analyze_dataset.py --data-dir chest_xray_dataset --z-threshold 2.5

  # Skip visualizations (faster)
  python analyze_dataset.py --data-dir chest_xray_dataset --no-plots
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='chest_xray_dataset',
                       help='Path to dataset directory (default: chest_xray_dataset)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory for results (default: analysis_results)')
    parser.add_argument('--z-threshold', type=float, default=3.0,
                       help='Z-score threshold for outlier detection (default: 3.0)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of images to sample per class (default: None = all images)')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run analysis on all images (ignores --sample-size)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots (faster)')
    
    args = parser.parse_args()
    
    # Override sample size for full analysis
    sample_size = None if args.full_analysis else args.sample_size
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        z_threshold=args.z_threshold,
        sample_size=sample_size
    )
    
    # Run analysis
    analyzer.run_full_analysis(generate_plots=not args.no_plots)


if __name__ == '__main__':
    main()
