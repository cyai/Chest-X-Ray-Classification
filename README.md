# Chest X-Ray Classification CNN

A deep learning model for classifying chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. This project implements a custom CNN architecture from scratch, optimized for GPU training on AWS g5.2xlarge instances.

## 📊 Dataset

- **Source**: [Kaggle - Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)
- **Classes**: 3 (Normal, Pneumonia, Tuberculosis)
- **Image Size**: 224x224 pixels
- **Total Samples**: 2,569 test images

## 🏗️ Model Architecture

### Custom CNN Architecture
- **5 Convolutional Blocks** with increasing depth (64 → 128 → 256 → 512 → 512)
- **Total Parameters**: ~40 million
- **Input**: 3-channel RGB images (224x224)
- **Output**: 3-class classification

#### Architecture Details:
```
Conv Block 1: 64 filters  → MaxPool → Dropout(0.25)
Conv Block 2: 128 filters → MaxPool → Dropout(0.25)
Conv Block 3: 256 filters → MaxPool → Dropout(0.30)
Conv Block 4: 512 filters → MaxPool → Dropout(0.30)
Conv Block 5: 512 filters → MaxPool → Dropout(0.30)
Adaptive Pooling: 7x7
Fully Connected: 25,088 → 4,096 → 2,048 → 3
```

### Key Features:
- **Batch Normalization** after each convolutional layer
- **ReLU Activation** functions
- **Dropout Regularization** (0.25-0.5)
- **Kaiming He Initialization** for convolutional layers

## ⚙️ Training Configuration

### Optimizer & Loss
- **Optimizer**: Adam
  - Learning Rate: 0.001
  - Weight Decay: 1e-4 (L2 regularization)
- **Loss Function**: CrossEntropyLoss
- **LR Scheduler**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 5 epochs

### Hyperparameters
- **Batch Size**: 64 (optimized for 24GB GPU)
- **Epochs**: 50 (with early stopping)
- **Early Stopping Patience**: 10 epochs
- **Image Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### GPU Optimizations
- **Mixed Precision Training** (FP16)
- **Number of Workers**: 8 (multi-process data loading)
- **Pin Memory**: Enabled
- **Persistent Workers**: Enabled
- **Prefetch Factor**: 4

### Data Augmentation (Training Only)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Random Affine Translation (0.1)
- Color Jitter (brightness=0.2, contrast=0.2)
- Random Resized Crop (scale=0.8-1.0)

## 📈 Training Results

### Training History

![Training History](training_history.png)

The model was trained for 50+ epochs with the following progression:
- Training and validation loss steadily decreased
- Training accuracy reached ~75%
- Validation accuracy plateated around ~75%
- Learning rate was reduced automatically when validation loss plateaued

### Performance Metrics

**Test Accuracy**: **75.24%**

#### Per-Class Performance:

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Normal       | 0.6157    | 0.8573 | 0.7167   | 925     |
| Pneumonia    | 0.8253    | 0.9207 | 0.8704   | 580     |
| Tuberculosis | 0.9558    | 0.5695 | 0.7138   | 1,064   |
| **Weighted Avg** | **0.8039** | **0.7524** | **0.7502** | **2,569** |

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

**Per-Class Accuracy**:
- **Normal**: 85.73%
- **Pneumonia**: 92.07%
- **Tuberculosis**: 56.95%

### Analysis:
- ✅ **Pneumonia detection** performs excellently (92% recall)
- ✅ **Normal cases** have high recall (86%)
- ⚠️ **Tuberculosis** has high precision (96%) but lower recall (57%), indicating conservative predictions
- Main confusion: Tuberculosis often misclassified as Normal (451 cases)

## 🖼️ Sample Predictions

![Sample Predictions](sample_predictions.png)

The visualization shows model predictions on test samples with:
- ✅ **Green** titles: Correct predictions
- ❌ **Red** titles: Incorrect predictions

## 📁 Project Structure

```
.
├── train-model.ipynb          # Main training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── best_model.pth             # Best model checkpoint
├── chest_xray_cnn_final.pth   # Final model with metadata
├── training_history.png        # Training curves
├── confusion_matrix.png        # Confusion matrix visualization
├── sample_predictions.png      # Sample prediction results
├── classification_report.txt   # Detailed metrics report
└── chest_xray_dataset/        # Dataset directory
    ├── train/
    ├── val/
    └── test/
```

## 🚀 Installation & Usage

### Prerequisites
```bash
# Python 3.8+
# CUDA 11.8+ (for GPU support)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup Kaggle API
1. Go to https://www.kaggle.com/settings/account
2. Create API token and download `kaggle.json`
3. Place it in `~/.kaggle/kaggle.json`

### Run Training
Open and run `train-model.ipynb` in Jupyter:
```bash
jupyter notebook train-model.ipynb
```

Or use JupyterLab:
```bash
jupyter lab train-model.ipynb
```

### Load Trained Model
```python
import torch
from train_model import ChestXRayCNN

# Load model
checkpoint = torch.load('best_model.pth')
model = ChestXRayCNN(num_classes=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 💻 Hardware Requirements

- **Recommended**: AWS g5.2xlarge or equivalent
  - GPU: NVIDIA A10G (24GB VRAM)
  - vCPUs: 8
  - RAM: 32GB
- **Minimum**: Any CUDA-capable GPU with 8GB+ VRAM

## 📊 Training Time

- **Per Epoch**: ~2-5 minutes
- **Total Training**: ~1-2 hours (with early stopping)
- **GPU Memory Usage**: ~10-15GB

## 🔬 Key Technologies

- **Framework**: PyTorch 2.0+
- **Image Processing**: TorchVision, PIL
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn
- **Progress Tracking**: tqdm

## 📝 Detailed Classification Report

```
              precision    recall  f1-score   support

      normal     0.6157    0.8573    0.7167       925
   pneumonia     0.8253    0.9207    0.8704       580
tuberculosis     0.9558    0.5695    0.7138      1064

    accuracy                         0.7524      2569
   macro avg     0.7990    0.7825    0.7670      2569
weighted avg     0.8039    0.7524    0.7502      2569
```

## 🎯 Future Improvements

1. **Class Imbalance Handling**: Implement weighted loss or focal loss
2. **Data Augmentation**: Advanced techniques like mixup or cutmix
3. **Architecture**: Try deeper networks or attention mechanisms
4. **Ensemble Methods**: Combine multiple models
5. **Transfer Learning**: Fine-tune pre-trained models (ResNet, EfficientNet)
6. **Tuberculosis Recall**: Focus on improving recall for TB cases

## 📚 References

- Dataset: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)
- Medical imaging best practices
- PyTorch documentation

## 📄 License

This project is for educational and research purposes.

## 👥 Author

Created as a deep learning project for chest X-ray classification.

---

**Note**: This model is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.

nohup python train_rnn.py > log_train_rnn.log 2>&1 &
kill %1

nohup python train_all_models.py > train_all_models.log 2>&1 &

kill %1