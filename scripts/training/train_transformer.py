# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# %%
# Configuration
class Config:
    # Paths
    data_dir = "/home/ubuntu/dl/chest_xray_dataset"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Model parameters
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2  # 196
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    mlp_ratio = 4
    dropout = 0.1
    num_classes = 3

    # Training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 7

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
print(f"Image size: {config.img_size}x{config.img_size}")
print(f"Patch size: {config.patch_size}x{config.patch_size}")
print(f"Number of patches: {config.num_patches}")
print(f"Embedding dimension: {config.embed_dim}")


# %%
# Dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["normal", "pneumonia", "tuberculosis"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# %%
# Data transforms
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

val_transform = transforms.Compose(
    [
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = ChestXRayDataset(config.train_dir, transform=train_transform)
val_dataset = ChestXRayDataset(config.val_dir, transform=val_transform)
test_dataset = ChestXRayDataset(config.test_dir, transform=val_transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# %%
# Patch Embedding Layer
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# %%
# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


# %%
# MLP Block
class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# %%
# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# %%
# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=3,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take class token
        x = self.head(cls_token_final)

        return x


# %%
# Initialize model
model = VisionTransformer(
    img_size=config.img_size,
    patch_size=config.patch_size,
    in_channels=3,
    num_classes=config.num_classes,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    mlp_ratio=config.mlp_ratio,
    dropout=config.dropout,
).to(config.device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nModel architecture:")
print(model)

# %%
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

print("Training setup complete")
print(f"Loss function: CrossEntropyLoss")
print(
    f"Optimizer: AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})"
)
print(f"Scheduler: CosineAnnealingLR")


# %%
# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"}
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# %%
# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# %%
# Training loop with early stopping
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

best_val_acc = 0.0
patience_counter = 0

print(f"Starting training for {config.num_epochs} epochs...")
print(f"Early stopping patience: {config.patience} epochs\n")

for epoch in range(config.num_epochs):
    print(f"Epoch [{epoch+1}/{config.num_epochs}]")

    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, config.device
    )

    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)

    # Update scheduler
    scheduler.step()

    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_vit_model.pth")
        print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)\n")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= config.patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"\nTraining completed!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# %%
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history["train_loss"], label="Train Loss", marker="o")
ax1.plot(history["val_loss"], label="Val Loss", marker="s")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(history["train_acc"], label="Train Acc", marker="o")
ax2.plot(history["val_acc"], label="Val Acc", marker="s")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("vit_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

print("Training history plot saved as 'vit_training_history.png'")

# %%
# Load best model for evaluation
model.load_state_dict(torch.load("best_vit_model.pth"))
print("Loaded best model for evaluation")


# %%
# Test evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# Get predictions
y_true, y_pred = evaluate_model(model, test_loader, config.device)

# Calculate accuracy
test_acc = 100 * np.sum(y_true == y_pred) / len(y_true)
print(f"\nTest Accuracy: {test_acc:.2f}%")

# %%
# Classification report
class_names = ["Normal", "Pneumonia", "Tuberculosis"]
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

# Save report
with open("vit_classification_report.txt", "w") as f:
    f.write(report)
print("Classification report saved to 'vit_classification_report.txt'")

# %%
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

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
plt.title("Confusion Matrix - Vision Transformer", fontsize=16, pad=20)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig("vit_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("Confusion matrix saved as 'vit_confusion_matrix.png'")

# %%
# Save final model with metadata
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "img_size": config.img_size,
            "patch_size": config.patch_size,
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "num_classes": config.num_classes,
        },
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "class_names": class_names,
    },
    "chest_xray_vit_final.pth",
)

print("Final model saved as 'chest_xray_vit_final.pth'")

# %% [markdown]
# ## Model Summary
#
# **Vision Transformer Architecture:**
# - **Patch Size**: 16×16 pixels (196 patches per image)
# - **Embedding Dimension**: 512
# - **Transformer Layers**: 6
# - **Attention Heads**: 8 per layer
# - **MLP Ratio**: 4 (hidden dim = 2048)
# - **Dropout**: 0.1
#
# **Key Components:**
# 1. **Patch Embedding**: Converts image into sequence of patches
# 2. **CLS Token**: Learnable classification token
# 3. **Positional Encoding**: Adds spatial information
# 4. **Multi-Head Self-Attention**: Captures global dependencies
# 5. **Feed-Forward Networks**: Per-token processing
# 6. **Layer Normalization**: Stabilizes training
#
# **Training Configuration:**
# - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR
# - Batch Size: 32
# - Early Stopping: Patience 7 epochs
# - Data Augmentation: Random flips, rotation, color jitter
