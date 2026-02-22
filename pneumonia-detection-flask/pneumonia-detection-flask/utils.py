"""
Utility functions for Pneumonia Detection project.
Handles preprocessing, model helpers, and Grad-CAM visualization.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path helpers (avoid hardcoded paths)
# ---------------------------------------------------------------------------

def get_project_root():
    """Return project root directory (parent of this file)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_model_path():
    """Return path to saved model weights."""
    root = get_project_root()
    return os.path.join(root, 'model', 'pneumonia_model.pth')


def get_uploads_dir():
    """Return path to uploads directory."""
    root = get_project_root()
    return os.path.join(root, 'static', 'uploads')


# ---------------------------------------------------------------------------
# Image preprocessing (aligned with training)
# ---------------------------------------------------------------------------

# ImageNet normalization for ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def get_inference_transform():
    """
    Transform for inference: resize, to tensor, normalize.
    Must match preprocessing used during training.
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_path_or_pil):
    """
    Preprocess image for model input.
    Args:
        image_path_or_pil: path (str) or PIL Image
    Returns:
        tensor: (1, 3, 224, 224)
    """
    transform = get_inference_transform()
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert('RGB')
    else:
        img = image_path_or_pil.convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Model definition (must match train_model.py)
# ---------------------------------------------------------------------------

def build_resnet50_pneumonia(num_classes=2, pretrained=False):
    """
    Build ResNet50 with replaced classifier for binary pneumonia classification.
    """
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_pneumonia_model(device=None):
    """
    Load trained model from model/pneumonia_model.pth.
    Returns model on correct device, or None if file not found.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = get_model_path()
    if not os.path.isfile(path):
        return None
    model = build_resnet50_pneumonia(num_classes=2, pretrained=False)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for ResNet50: hook on last conv layer (layer4).
    """

    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _get_target_layer(self):
        return getattr(self.model, self.target_layer_name)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        target = self._get_target_layer()
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def generate(self, input_tensor, target_class_idx=None):
        """
        Generate Grad-CAM heatmap.
        input_tensor: (1, 3, H, W)
        target_class_idx: class index to backprop (default: predicted class)
        Returns: heatmap (numpy 2D), predicted class index
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        # weights = mean of gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3))
        # cam = weighted sum of activations
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        # resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class_idx


def create_gradcam_heatmap_image(model, input_tensor, original_image_path, output_path, device):
    """
    Generate Grad-CAM overlay and save as image.
    original_image_path: path to original image for overlay
    output_path: where to save the heatmap image (e.g. static/uploads/gradcam_xxx.png)
    """
    grad_cam = GradCAM(model, target_layer_name='layer4')
    input_tensor = input_tensor.to(device)
    heatmap, _ = grad_cam.generate(input_tensor)

    # Load original image for overlay
    img = cv2.imread(original_image_path)
    if img is None:
        img = np.array(Image.open(original_image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    return output_path
