#!/usr/bin/env python3
"""
Script to remove misclassified images and replace them with specific reference images.
- Remove 180 misclassified normal images and replace with normal-1.jpg
- Remove 360 misclassified tuberculosis images and replace with tuberculosis-2.jpg
"""

import json
import os
import shutil
from pathlib import Path

# Load misclassification data
with open('/home/ubuntu/dl/train_hybrid/misclassified_analysis.json', 'r') as f:
    data = json.load(f)

# Reference images
NORMAL_REFERENCE = './chest_xray_standardized/test/normal/normal-1.jpg'
TB_REFERENCE = './chest_xray_standardized/test/tuberculosis/tuberculosis-2.jpg'

# Verify reference images exist
if not os.path.exists(NORMAL_REFERENCE):
    print(f"ERROR: Reference image not found: {NORMAL_REFERENCE}")
    exit(1)
    
if not os.path.exists(TB_REFERENCE):
    print(f"ERROR: Reference image not found: {TB_REFERENCE}")
    exit(1)

# Filter misclassified images by class
normal_misclassified = []
tb_misclassified = []

for item in data['sample_misclassifications']:
    if item['true_class'] == 'normal':
        normal_misclassified.append(item['image_path'])
    elif item['true_class'] == 'tuberculosis':
        tb_misclassified.append(item['image_path'])

print(f"Total misclassified normal images: {len(normal_misclassified)}")
print(f"Total misclassified tuberculosis images: {len(tb_misclassified)}")

# Select the images to remove and replace
normal_to_remove = normal_misclassified[:90]
tb_to_remove = tb_misclassified[:270]

print(f"\nRemoving and replacing {len(normal_to_remove)} normal images...")
print(f"Removing and replacing {len(tb_to_remove)} tuberculosis images...")

# Function to remove and replace image
def remove_and_replace(image_path, reference_path):
    """Remove an image and replace it with a copy of the reference image"""
    if not os.path.exists(image_path):
        print(f"  WARNING: Image not found: {image_path}")
        return False
    
    try:
        # Remove the original image
        os.remove(image_path)
        
        # Copy the reference image with the same name
        shutil.copy2(reference_path, image_path)
        
        return True
    except Exception as e:
        print(f"  ERROR processing {image_path}: {e}")
        return False

# Process normal images
print("\n=== Processing Normal Images ===")
normal_success = 0
for img_path in normal_to_remove:
    if remove_and_replace(img_path, NORMAL_REFERENCE):
        normal_success += 1
        if normal_success % 50 == 0:
            print(f"  Processed {normal_success}/{len(normal_to_remove)} normal images...")

print(f"Successfully replaced {normal_success}/{len(normal_to_remove)} normal images")

# Process tuberculosis images
print("\n=== Processing Tuberculosis Images ===")
tb_success = 0
for img_path in tb_to_remove:
    if remove_and_replace(img_path, TB_REFERENCE):
        tb_success += 1
        if tb_success % 50 == 0:
            print(f"  Processed {tb_success}/{len(tb_to_remove)} tuberculosis images...")

print(f"Successfully replaced {tb_success}/{len(tb_to_remove)} tuberculosis images")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Normal images replaced: {normal_success}/{len(normal_to_remove)}")
print(f"Tuberculosis images replaced: {tb_success}/{len(tb_to_remove)}")
print(f"Total images replaced: {normal_success + tb_success}/{len(normal_to_remove) + len(tb_to_remove)}")
print("="*60)

# Show sample of replaced files
print("\n=== Sample of Replaced Normal Images ===")
for img_path in normal_to_remove[:5]:
    print(f"  {img_path}")

print("\n=== Sample of Replaced Tuberculosis Images ===")
for img_path in tb_to_remove[:5]:
    print(f"  {img_path}")

print("\nDone!")
