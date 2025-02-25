import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm  # Progress bar
from PIL import Image

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for 2 classes (Healthy, Diseased)
model = model.to(device)
model.eval()

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook for activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)  # Proper hook

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self):
        # Compute Grad-CAM
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.clone()  # Fix in-place operation

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]

        cam = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)  # ReLU (only positive influences)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize safely

        return cam

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Set dataset path
dataset_path = "/home/bruhjeshh/Coding /anothatry/dataset"  # Adjust path

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

# Get class names
class_names = dataset.classes  # ['healthy', 'unhealthy']
print(f"Classes: {class_names}")

# Grad-CAM Generator
cam_generator = GradCAM(model, model.layer4[-1])

# Function to Generate and Save Grad-CAM
def generate_grad_cam(model, image, label, epoch):
    model.zero_grad()
    
    # Convert to batch format
    image = image.unsqueeze(0).to(device)
    
    # Forward pass
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    
    # Get class names
    pred_class_name = class_names[pred_class]
    actual_class_name = class_names[label]

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Generate Grad-CAM
    grad_cam = cam_generator.generate_cam()
    
    # Load original image
    img = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_result = np.clip(heatmap * 0.4 + img, 0, 1)  # Blend heatmap with image

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\nActual: {actual_class_name}")
    axes[0].axis("off")

    axes[1].imshow(cam_result)
    axes[1].set_title(f"Grad-CAM\nPredicted: {pred_class_name}")
    axes[1].axis("off")

    # Create output directory
    save_dir = f"grad_cam_outputs/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    # Save image
    save_path = os.path.join(save_dir, f"grad_cam_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved Grad-CAM at {save_path}")

# Training Loop with Progress Bar
num_epochs = 300
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    with tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=running_loss / total, acc=correct / total)

   
    print(f"\n🖼 Generating and Saving Grad-CAM at epoch {epoch}...")
    generate_grad_cam(model, images[0], labels[0].item(), epoch)

print("✅ Training complete!")
