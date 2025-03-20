import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset_path = "/home/bruhjeshh/Coding /PE-2/Resnet50+GradCam/dataset1"  # Update path
dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
class_names = dataset.classes

# Verify number of classes
num_classes = len(dataset.classes)
print(f"Number of classes in dataset: {num_classes}")

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the correct number of classes
model = model.to(device)
model.eval()

# Grad-CAM class (improved for better focus on diseased regions)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self):
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]
        cam = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Grad-CAM Generator
cam_generator = GradCAM(model, model.layer4[-1])

# Function to Generate and Save Grad-CAM with Proper Overlay

def generate_grad_cam(model, image, label, epoch):
    model.zero_grad()
    image = image.unsqueeze(0).to(device)
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    pred_class_name = class_names[pred_class]
    actual_class_name = class_names[label]
    output[0, pred_class].backward()
    grad_cam = cam_generator.generate_cam()
    
    # Convert tensor image to numpy
    img = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Stack original and Grad-CAM output
    stacked_output = np.vstack((img, blended))
    
    # Create a blank white space for text
    text_space = np.ones((80, stacked_output.shape[1], 3), dtype=np.uint8) * 255  # Increased height for text
    
    # Prepare text
    text = f"Predicted: {pred_class_name} | Actual: {actual_class_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 0, 0)
    
    # Get text size to check if it fits
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # If text is too wide, split into two lines
    if text_width > stacked_output.shape[1] - 20:  # Leave some margin
        # Split text into two lines
        words = text.split()
        line1 = " ".join(words[:len(words)//2])
        line2 = " ".join(words[len(words)//2:])
        
        # Calculate positions for two lines
        (line1_width, line1_height), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (line2_width, line2_height), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        
        # Center text vertically and horizontally
        y1 = int((text_space.shape[0] - (line1_height + line2_height + 10)) / 2) + line1_height
        y2 = y1 + line2_height + 10
        x1 = int((text_space.shape[1] - line1_width) / 2)
        x2 = int((text_space.shape[1] - line2_width) / 2)
        
        # Add text to the image
        cv2.putText(text_space, line1, (x1, y1), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(text_space, line2, (x2, y2), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        # Center text vertically and horizontally
        x = int((text_space.shape[1] - text_width) / 2)
        y = int((text_space.shape[0] + text_height) / 2)
        cv2.putText(text_space, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Combine image and text
    final_output = np.vstack((stacked_output, text_space))
    
    save_dir = f"grad_cam_outputs/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"grad_cam_epoch_{epoch}.png")
    cv2.imwrite(save_path, final_output)
    print(f"✅ Saved Grad-CAM at {save_path}")

# Training Loop
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
        for images, labels in pbar:
            assert labels.min() >= 0 and labels.max() < num_classes, "Invalid labels detected!"
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
