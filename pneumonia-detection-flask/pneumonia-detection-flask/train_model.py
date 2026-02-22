"""
Training script for Pneumonia Detection model.
Uses Kaggle Chest X-Ray dataset with ResNet50 transfer learning.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use same constants as utils for consistency
from utils import get_project_root, get_model_path, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set True for quick testing: small data subset, 2 epochs, reduced batch size.
# Or pass --debug on the command line.
DEBUG_MODE = False

BATCH_SIZE = 32
NUM_EPOCHS = 20
INITIAL_LR = 1e-3
PATIENCE_EARLY_STOP = 5
NUM_WORKERS = 0  # Set > 0 on Linux/Mac if desired
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Debug-mode limits (used when DEBUG_MODE=True or --debug)
DEBUG_TRAIN_SAMPLES = 200
DEBUG_VAL_SAMPLES = 50
DEBUG_EPOCHS = 2
DEBUG_BATCH_SIZE = 8


def get_dataset_path():
    """Resolve dataset root: same folder as project or 'dataset' inside project."""
    root = get_project_root()
    for candidate in [os.path.join(root, 'dataset'), root]:
        train_normal = os.path.join(candidate, 'train', 'NORMAL')
        if os.path.isdir(train_normal):
            return candidate
    return os.path.join(root, 'dataset')


# ---------------------------------------------------------------------------
# Data augmentation and transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    """Training transform with augmentation."""
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform():
    """Validation/Test transform: no augmentation."""
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def create_dataloaders(dataset_root, batch_size=32, max_train=None, max_val=None):
    """
    Create train, val, test DataLoaders.
    Dataset structure: dataset_root/train|val|test/NORMAL|PNEUMONIA/
    If max_train/max_val are set, use a random subset (for debug mode).
    """
    train_transform = get_train_transform()
    val_transform = get_val_test_transform()

    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    test_dir = os.path.join(dataset_root, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    if max_train is not None and len(train_dataset) > max_train:
        indices = torch.randperm(len(train_dataset))[:max_train].tolist()
        train_dataset = Subset(train_dataset, indices)
    if max_val is not None and len(val_dataset) > max_val:
        indices = torch.randperm(len(val_dataset))[:max_val].tolist()
        val_dataset = Subset(val_dataset, indices)

    # For class_to_idx we need the underlying ImageFolder when using Subset
    base_train = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    class_to_idx = base_train.class_to_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader, class_to_idx


# ---------------------------------------------------------------------------
# Model setup (transfer learning)
# ---------------------------------------------------------------------------

def build_model(num_classes=2, freeze_backbone=True):
    """
    ResNet50 with ImageNet weights; replace final layer.
    Optionally freeze backbone for initial training.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = True
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc='Train', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=loss.item())
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1 (binary)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def plot_training_curves(history, save_dir):
    """Save loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection model')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset root (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: 32, or 8 in debug)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: 20, or 2 in debug)')
    parser.add_argument('--lr', type=float, default=INITIAL_LR)
    parser.add_argument('--patience', type=int, default=PATIENCE_EARLY_STOP)
    parser.add_argument('--unfreeze', action='store_true', help='Train with backbone unfrozen from start')
    parser.add_argument('--save_plots', type=str, default=None, help='Directory to save training plots')
    parser.add_argument('--debug', action='store_true', help='Debug mode: 200 train, 50 val, 2 epochs, batch 8')
    args = parser.parse_args()

    debug = args.debug or DEBUG_MODE
    if debug:
        print('*** DEBUG MODE IS ACTIVE ***')
        print(f'  Using subset: {DEBUG_TRAIN_SAMPLES} train, {DEBUG_VAL_SAMPLES} val samples')
        print(f'  Epochs: {DEBUG_EPOCHS}, Batch size: {DEBUG_BATCH_SIZE}')
        print('***')

    dataset_root = args.data or get_dataset_path()
    if not os.path.isdir(os.path.join(dataset_root, 'train')):
        raise FileNotFoundError(
            f'Dataset not found at {dataset_root}. '
            'Please place the Kaggle Chest X-Ray dataset with train/val/test and NORMAL/PNEUMONIA folders.'
        )

    batch_size = args.batch_size if args.batch_size is not None else (DEBUG_BATCH_SIZE if debug else BATCH_SIZE)
    epochs = args.epochs if args.epochs is not None else (DEBUG_EPOCHS if debug else NUM_EPOCHS)
    max_train = DEBUG_TRAIN_SAMPLES if debug else None
    max_val = DEBUG_VAL_SAMPLES if debug else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        dataset_root, batch_size=batch_size, max_train=max_train, max_val=max_val
    )
    print(f'Classes: {class_to_idx}')

    model = build_model(num_classes=2, freeze_backbone=not args.unfreeze)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True,
    )

    model_dir = os.path.dirname(get_model_path())
    os.makedirs(model_dir, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        print(f'\n--- Epoch {epoch}/{epochs} ---')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = get_model_path()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
            }, save_path)
            print(f'  -> Best model saved to {save_path}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping after {args.patience} epochs without improvement.')
                break

        # Optionally unfreeze backbone after a few epochs (skip in debug with only 2 epochs)
        if epoch == 3 and not args.unfreeze and epochs > 3:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            print('Backbone unfrozen; fine-tuning with lower LR.')

    # Load best model and evaluate on test set
    checkpoint = torch.load(get_model_path(), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, y_true, y_pred = validate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)
    print('\n--- Test Set Results ---')
    print(f'Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print(f'F1 Score:  {metrics["f1"]:.4f}')
    print('Confusion Matrix:')
    print(metrics['confusion_matrix'])

    # Save plots
    plot_dir = args.save_plots or os.path.join(get_project_root(), 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_curves(history, plot_dir)
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, os.path.join(plot_dir, 'confusion_matrix.png'))
    print(f'Plots saved to {plot_dir}')


if __name__ == '__main__':
    main()
