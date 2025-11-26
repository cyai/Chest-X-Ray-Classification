# %% [markdown]
# ## 1. Import Libraries and Setup

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# %% [markdown]
# ## 2. Configuration and Hyperparameters


# %%
# Configuration
class Config:
    # Paths
    train_dir = "./chest_xray_standardized/train"
    val_dir = "./chest_xray_standardized/val"
    test_dir = "./chest_xray_standardized/test"

    # Model parameters
    num_classes = 3  # Normal, Pneumonia, Tuberculosis

    # Image parameters
    img_size = 224

    # RNN-specific parameters
    rnn_type = "LSTM"  # 'LSTM' or 'GRU'
    hidden_size = 512
    num_layers = 3
    bidirectional = True
    dropout_rnn = 0.3

    # Sequence parameters (treat image as sequence of rows)
    sequence_length = 224  # Number of rows
    input_size = 224 * 3  # Each row has 224 pixels × 3 channels

    # Training hyperparameters
    batch_size = 32  # Reduced for RNN memory requirements
    num_epochs = 50
    learning_rate = 0.0005
    weight_decay = 1e-4

    # GPU optimization
    num_workers = 8
    pin_memory = True
    prefetch_factor = 4

    # Training settings
    early_stopping_patience = 15
    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
print(f"Training on: {config.device}")
print(f"RNN Type: {config.rnn_type}")
print(f"Hidden Size: {config.hidden_size}")
print(f"Num Layers: {config.num_layers}")
print(f"Bidirectional: {config.bidirectional}")
print(f"Batch size: {config.batch_size}")
print(f"Sequence length: {config.sequence_length}")
print(f"Input size per sequence step: {config.input_size}")

# %% [markdown]
# ## 3. Data Preprocessing
#
# For RNN processing, we'll convert images into sequences. Each image row becomes a sequence step.

# %%
# Data augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Validation and test transforms
val_test_transform = transforms.Compose(
    [
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print("Transforms defined successfully!")

# %% [markdown]
# ## 4. Custom Dataset Wrapper for RNN
#
# This wrapper reshapes images into sequences for RNN processing.


# %%
class RNNImageDataset(Dataset):
    """Wrapper to convert images into sequences for RNN"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # img shape: (3, 224, 224)

        # Reshape to sequence: (seq_len, input_size)
        # Each row of the image becomes a sequence step
        # (3, 224, 224) -> (224, 224*3) = (seq_len, features)
        img = img.permute(1, 2, 0)  # (224, 224, 3)
        img = img.reshape(config.sequence_length, config.input_size)  # (224, 672)

        return img, label


print("RNN Dataset wrapper defined!")

# %% [markdown]
# ## 5. Load Datasets

# %%
# Load base datasets
train_base = ImageFolder(root=config.train_dir, transform=train_transform)
val_base = ImageFolder(root=config.val_dir, transform=val_test_transform)
test_base = ImageFolder(root=config.test_dir, transform=val_test_transform)

# Wrap with RNN dataset
train_dataset = RNNImageDataset(train_base)
val_dataset = RNNImageDataset(val_base)
test_dataset = RNNImageDataset(test_base)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    prefetch_factor=config.prefetch_factor,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    prefetch_factor=config.prefetch_factor,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)

# Get class names
class_names = train_base.classes
print(f"Classes: {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Train batches per epoch: {len(train_loader)}")

# Test data shape
sample_img, sample_label = train_dataset[0]
print(f"\nSample sequence shape: {sample_img.shape}")
print(f"Expected: ({config.sequence_length}, {config.input_size})")

# %% [markdown]
# ## 6. RNN-based Model Architecture
#
# This model processes images as sequences using LSTM/GRU layers.


# %%
class ChestXRayRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        rnn_type="LSTM",
        bidirectional=True,
        dropout=0.3,
    ):
        super(ChestXRayRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        # Input projection layer to reduce dimensionality
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2)
        )

        # RNN layers
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

        # Calculate RNN output size
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)  # (batch, seq_len, 512)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size*2 if bidirectional)

        # Attention mechanism
        attention_weights = self.attention(rnn_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.sum(
            attention_weights * rnn_out, dim=1
        )  # (batch, hidden_size*2)

        # Classification
        output = self.classifier(context)  # (batch, num_classes)

        return output


# Create model
model = ChestXRayRNN(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    num_classes=config.num_classes,
    rnn_type=config.rnn_type,
    bidirectional=config.bidirectional,
    dropout=config.dropout_rnn,
).to(config.device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Architecture: {config.rnn_type}-based Classifier")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model moved to: {next(model.parameters()).device}")
print(f"\nModel Summary:")
print(model)

# %% [markdown]
# ## 7. Loss Function and Optimizer

# %%
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.lr_scheduler_factor,
    patience=config.lr_scheduler_patience,
    # verbose=True
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

print("Optimizer: Adam")
print(f"Initial learning rate: {config.learning_rate}")
print(f"Weight decay: {config.weight_decay}")
print("Mixed precision training: Enabled")

# %% [markdown]
# ## 8. Training and Validation Functions


# %%
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping for RNN stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


print("Training and validation functions defined!")

# %% [markdown]
# ## 9. Training Loop with Early Stopping

# %%
# Training history
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

# Early stopping variables
best_val_loss = float("inf")
best_val_acc = 0.0
patience_counter = 0
best_model_path = "rnn_train/best_rnn_model.pth"

print(f"Starting training for {config.num_epochs} epochs...")
print(f"Training on: {config.device}")
print("=" * 70)

start_time = time.time()

for epoch in range(config.num_epochs):
    epoch_start = time.time()

    print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
    print("-" * 70)

    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler, config.device
    )

    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)

    # Update learning rate
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    epoch_time = time.time() - epoch_start

    # Print epoch summary
    print(f"\nEpoch Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            best_model_path,
        )
        print(
            f"  ✓ Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)"
        )
    else:
        patience_counter += 1
        print(f"  Early stopping: {patience_counter}/{config.early_stopping_patience}")

    # Early stopping
    if patience_counter >= config.early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
        break

    # Clear cache periodically
    if (epoch + 1) % 5 == 0:
        torch.cuda.empty_cache()

total_time = time.time() - start_time
print("\n" + "=" * 70)
print(f"Training completed in {total_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("=" * 70)

# %% [markdown]
# ## 10. Plot Training History

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot loss
axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("RNN Model: Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot accuracy
axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("RNN Model: Training and Validation Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot learning rate
axes[2].plot(history["lr"], marker="o", color="green")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("RNN Model: Learning Rate Schedule")
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rnn_train/rnn_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

print("Training history plots saved!")

# %% [markdown]
# ## 11. Load Best Model and Evaluate on Test Set

# %%
# Load best model
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Best model loaded from epoch {checkpoint['epoch']+1}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

# Evaluate on test set
model.eval()
all_preds = []
all_labels = []
test_loss = 0.0

print("\nEvaluating on test set...")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_dataset)
test_acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# %% [markdown]
# ## 12. Classification Report

# %%
# Generate classification report
report = classification_report(
    all_labels, all_preds, target_names=class_names, digits=4
)
print("\nClassification Report:")
print("=" * 70)
print(report)

# Save report to file
with open("rnn_train/rnn_classification_report.txt", "w") as f:
    f.write("RNN Model Classification Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model Type: {config.rnn_type}\n")
    f.write(f"Hidden Size: {config.hidden_size}\n")
    f.write(f"Num Layers: {config.num_layers}\n")
    f.write(f"Bidirectional: {config.bidirectional}\n\n")
    f.write(report)
print("Classification report saved to 'rnn_train/rnn_classification_report.txt'")

# %% [markdown]
# ## 13. Confusion Matrix

# %%
# Generate and plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "Count"},
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("RNN Model - Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("rnn_train/rnn_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Calculate per-class accuracy
print("\nPer-class accuracy:")
for i, class_name in enumerate(class_names):
    class_acc = 100.0 * cm[i, i] / cm[i].sum()
    print(f"  {class_name}: {class_acc:.2f}%")

# %% [markdown]
# ## 14. Visualize Sample Predictions

# %%
# Get some test samples for visualization
# We need to get original images, not sequences
test_base_loader = DataLoader(test_base, batch_size=16, shuffle=True)


def denormalize(tensor):
    """Denormalize image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


# Get a batch
dataiter = iter(test_base_loader)
images, labels = next(dataiter)

# Convert to sequences for prediction
sequences = []
for img in images:
    img_seq = img.permute(1, 2, 0).reshape(config.sequence_length, config.input_size)
    sequences.append(img_seq)
sequences = torch.stack(sequences).to(config.device)

# Get predictions
model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast():
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)

# Plot sample predictions
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()

for idx in range(16):
    if idx < len(images):
        img = denormalize(images[idx])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true_label = class_names[labels[idx]]
        pred_label = class_names[predicted[idx].cpu()]

        color = "green" if labels[idx] == predicted[idx].cpu() else "red"

        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color,
            fontsize=10,
            fontweight="bold",
        )

plt.suptitle("RNN Model - Sample Predictions", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig("rnn_train/rnn_sample_predictions.png", dpi=300, bbox_inches="tight")
plt.show()

print("Sample predictions visualized!")

# %% [markdown]
# ## 15. Save Final Model

# %%
# Save complete model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": {
            "num_classes": config.num_classes,
            "img_size": config.img_size,
            "rnn_type": config.rnn_type,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "bidirectional": config.bidirectional,
            "input_size": config.input_size,
            "sequence_length": config.sequence_length,
        },
        "test_accuracy": test_acc,
        "history": history,
    },
    "rnn_train/chest_xray_rnn_final.pth",
)

print("Final model saved as 'rnn_train/chest_xray_rnn_final.pth'")
print("\nModel Summary:")
print(f"  Total parameters: {total_params:,}")
print(f"  Test accuracy: {test_acc:.2f}%")
print(f"  Classes: {class_names}")
print(f"  RNN Type: {config.rnn_type}")
print(f"  Hidden Size: {config.hidden_size}")
print(f"  Num Layers: {config.num_layers}")

# %% [markdown]
# ## 16. Model Comparison Summary

# %%
# Create comparison summary
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

# CNN results (from previous training)
cnn_results = {
    "Model": "Custom CNN",
    "Test Accuracy": "75.24%",
    "Normal Recall": "85.73%",
    "Pneumonia Recall": "92.07%",
    "Tuberculosis Recall": "56.95%",
}

# RNN results (current model)
rnn_results = {
    "Model": f"{config.rnn_type}-based",
    "Test Accuracy": f"{test_acc:.2f}%",
}

# Calculate per-class recall for RNN
from sklearn.metrics import recall_score

recalls = recall_score(all_labels, all_preds, average=None)
for i, class_name in enumerate(class_names):
    rnn_results[f"{class_name} Recall"] = f"{recalls[i]*100:.2f}%"

print("\nCNN Model (Previous):")
for key, value in cnn_results.items():
    print(f"  {key}: {value}")

print(f"\n{config.rnn_type} Model (Current):")
for key, value in rnn_results.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("\n1. Architecture Comparison:")
print("   - CNN: Spatial feature extraction, ~40M parameters")
print(
    f"   - RNN: Sequential processing with attention, ~{total_params/1e6:.1f}M parameters"
)
print("\n2. Processing Approach:")
print("   - CNN: Processes images as 2D spatial data")
print("   - RNN: Treats images as sequences of rows with temporal dependencies")
print("\n3. Computational Efficiency:")
print("   - CNN: Parallel convolution operations")
print("   - RNN: Sequential processing (slower but captures dependencies)")
print("\n4. Best Use Cases:")
print("   - CNN: Standard choice for image classification (spatial features)")
print("   - RNN: Experimental approach for sequential pattern analysis")
print("\n" + "=" * 70)

# %% [markdown]
# ## 17. GPU Utilization Summary

# %%
# GPU memory usage
if torch.cuda.is_available():
    print("GPU Memory Summary:")
    print("=" * 70)
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print("=" * 70)

    # Clear cache
    torch.cuda.empty_cache()
    print("GPU cache cleared!")

print("\n🎉 RNN Training pipeline completed successfully!")
print("\nGenerated files:")
print("  - rnn_train/best_rnn_model.pth (best model checkpoint)")
print("  - rnn_train/chest_xray_rnn_final.pth (final model with metadata)")
print("  - rnn_train/rnn_training_history.png (training curves)")
print("  - rnn_train/rnn_confusion_matrix.png (confusion matrix)")
print("  - rnn_train/rnn_sample_predictions.png (sample predictions)")
print("  - rnn_train/rnn_classification_report.txt (detailed metrics)")

# %% [markdown]
# ## 18. Final Recommendations
#
# ### Model Selection Guidelines:
#
# **Choose CNN if:**
# - You need standard, proven image classification performance
# - Spatial features (edges, textures, patterns) are most important
# - Faster training and inference are priorities
# - You want simpler model architecture
#
# **Choose RNN if:**
# - You want to explore sequential dependencies in image data
# - You're interested in experimental approaches
# - You have sufficient computational resources
# - You want attention mechanisms to identify important regions
#
# ### Hybrid Approach:
# Consider combining both:
# - Use CNN for feature extraction
# - Feed CNN features into RNN for sequential modeling
# - This is the basis for advanced architectures like CNN-LSTM
#
# ### Next Steps:
# 1. Try transfer learning with pre-trained models (ResNet, EfficientNet)
# 2. Implement ensemble methods combining multiple models
# 3. Use attention mechanisms (Vision Transformers)
# 4. Address class imbalance with weighted loss or sampling strategies
# 5. Collect more data, especially for Tuberculosis class
