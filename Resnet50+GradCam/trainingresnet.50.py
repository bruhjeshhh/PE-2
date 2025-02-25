import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from torchvision.utils import save_image

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load ResNet-50 (pretrained)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
model = model.to(device)

# ✅ Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Load Dataset
dataset_path = "/home/bruhjeshh/Coding /anothatry/dataset"  # Updated path
train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Hook to Extract Activations & Gradients
activations = None
gradients = None

def hook_fn(module, input, output):
    global activations
    activations = output

def backward_hook_fn(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# ✅ Attach Hooks to Last Convolutional Layer (ResNet Layer4)
model.layer4.register_forward_hook(hook_fn)
model.layer4.register_backward_hook(backward_hook_fn)

# ✅ Function to Generate & Save Grad-CAM
def generate_grad_cam(model, image, class_idx, epoch, device):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # ✅ Forward Pass
    output = model(image)
    
    # ✅ Backprop to get gradients
    model.zero_grad()
    target = output[0, class_idx]
    target.backward()

    # ✅ Compute Grad-CAM
    global activations, gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).view(-1, 1, 1)
    weighted_activations = activations * pooled_gradients
    cam = torch.mean(weighted_activations, dim=1).squeeze().cpu().detach().numpy()

    # ✅ Normalize & Resize Grad-CAM
    cam = np.maximum(cam, 0)  # ReLU: Remove negatives
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0,1]
    cam = cv2.resize(cam, (224, 224))  # Resize to match image size

    # ✅ Save Original Image
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to numpy
    original_path = f"gradcam_results/epoch_{epoch}_original.png"
    plt.imsave(original_path, img_np)
    
    # ✅ Save Grad-CAM Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # Apply color
    superimposed = np.float32(heatmap) / 255 + img_np  # Overlay
    os.makedirs("gradcam_results", exist_ok=True)
    save_path = f"gradcam_results/epoch_{epoch}_gradcam.png"
    superimposed = np.clip(superimposed, 0, 1)  # 🔹 Ensure values are within [0,1]
    plt.imsave(save_path, superimposed)

    
    print(f"🖼 Grad-CAM & Original Image saved for epoch {epoch}")

# ✅ Training Loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # ✅ Progress Bar for Training
    with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ✅ Track Accuracy & Loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            t.set_postfix(loss=running_loss / len(train_loader), acc=100. * correct / total)

    print(f"Epoch {epoch}/{num_epochs} - Loss: {running_loss:.4f} - Acc: {100. * correct / total:.2f}%")

    # ✅ Generate Grad-CAM at every epoch
    if epoch%50==0:
      print(f"\n🖼 Generating Grad-CAM at epoch {epoch}...")
      generate_grad_cam(model, images[0], class_idx=1, epoch=epoch, device=device)

print("✅ Training Complete!")
