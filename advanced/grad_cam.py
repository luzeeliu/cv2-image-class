import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        GradCAM initialization
        Args:
            model: Neural network model
            target_layer: Layer to perform GradCAM on
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        # Register hooks to save gradients and features
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        """Hook to save the forward pass features"""
        self.features = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save the backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate the GradCAM heatmap
        Args:
            input_tensor: Preprocessed input tensor
            class_idx: Class index for which GradCAM is computed (optional)
        Returns:
            Normalized GradCAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)

        # Auto-select class with maximum score if not provided
        if class_idx is None:
            class_idx = torch.argmax(output).item()

        # Backward pass with one-hot vector
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)

        # Compute channel weights using global average pooling
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
        cam = torch.matmul(pooled_gradients, self.features.reshape(self.features.shape[1], -1))
        cam = cam.reshape(self.features.shape[2:])
        cam = F.relu(cam)

        # Normalize and resize heatmap to input dimensions
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam.cpu().numpy(), (input_tensor.shape[3], input_tensor.shape[2]))

        return cam


def overlay_heatmap(raw_image, cam, alpha=0.5):
    """
    Overlay GradCAM heatmap onto the original image
    Args:
        raw_image: Original image in BGR format (H, W, 3)
        cam: Normalized heatmap (H, W)
        alpha: Opacity level of the heatmap
    Returns:
        Image with heatmap overlay (BGR)
    """
    raw_image = cv2.resize(raw_image, (cam.shape[1], cam.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(raw_image, alpha, heatmap, 1 - alpha, 0)
    return superimposed


def apply_gradcam_on_image(model, image_bgr, label=None, device='cpu',
                           target_layer=None, preprocess_fn=None,
                           save_path=None):
    """
    Apply GradCAM visualization on a single image
    Args:
        model: Neural network model
        image_bgr: Original BGR image read by OpenCV
        label: Target class index (auto-inferred if None)
        device: Computation device (e.g., 'cpu' or 'cuda')
        target_layer: Model layer to analyze
        preprocess_fn: Image preprocessing function
        save_path: Optional path to save the resulting image
    Returns:
        Image with GradCAM visualization (BGR)
    """
    model.to(device)
    model.eval()

    gradcam = GradCAM(model, target_layer)

    # Image preprocessing
    input_tensor = preprocess_fn(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    input_tensor = input_tensor.unsqueeze(0).to(device).requires_grad_(True)

    # Generate CAM heatmap and overlay on the original image
    cam = gradcam.generate(input_tensor, class_idx=label)
    result = overlay_heatmap(image_bgr, cam)

    # Optionally save the resulting image
    if save_path:
        cv2.imwrite(save_path, result)
        print(f"Grad-CAM visualization saved at: {save_path}")

    return result
