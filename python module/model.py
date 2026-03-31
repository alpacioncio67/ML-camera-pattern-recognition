import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.onnx
from PIL import Image

# 1. LOADING OUR EXACT DATASET
class MultiTaskDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # 1. Reading csv using pandas
        self.anotaciones = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # 2. HashMap for label encoding
        self.forma_a_idx = {"cuadrado": 0, "circulo": 1, "rectangulo": 2, "triangulo": 3}
        self.color_a_idx = {"rojo": 0, "verde": 1, "azul": 2}

    def __len__(self):
        # Total images
        return len(self.anotaciones)

    def __getitem__(self, idx):
        # 1. Obtaining file name and full route
        img_name = self.anotaciones.iloc[idx, 0] # First column: file name
        img_path = f"{self.root_dir}/{img_name}"
        
        # 2. Load image with openCV
        image = cv2.imread(img_path)
        
        # Important: OpenCV loads in BGR, we need to use RGB for Pytorch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Change format for our transformations
        image = Image.fromarray(image)
        
        # 3. Obtaining text labels and changing to number (label encoding)
        forma_texto = self.anotaciones.iloc[idx, 1]
        color_texto = self.anotaciones.iloc[idx, 2]
        
        label_forma = self.forma_a_idx[forma_texto]
        label_color = self.color_a_idx[color_texto]
        
        # 4. Applying transformations
        if self.transform:
            image = self.transform(image)
            
        # We return our image ready to go, the label and the colour number
        return image, label_forma, label_color

# Basic transformations
transformaciones = transforms.Compose([
    transforms.ToTensor() # Change to tensor (C,H,W)
])

# Create dataset and dataloader
mi_dataset = MultiTaskDataset(csv_file="etiquetas.csv", root_dir="dataset_imagenes", transform=transformaciones)

# Loading images in batches of 32
dataloader = DataLoader(mi_dataset, batch_size=32, shuffle=True)


# 2. Network arquitecture
class ShapeColorCNN(nn.Module):
    def __init__(self, num_formas=4, num_colores=3):
        super(ShapeColorCNN, self).__init__()
        
        # --- Characteristics extractor ---
        # Entry: 3 channels (RGB) and 64x64 pixels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image to 32x32
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x15
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 xd
        )
        
        # Flattering our matrix
        # Size = 64 filters * 9 height * 8 width = 4096
        self.flatten = nn.Flatten()
        
        # HEADER 1: Form prediction
        self.head_forma = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_formas) # Output:  4 values: our forms
        )
        
        # HEADER 2: Color prediction
        self.head_color = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_colores) # Output: 3 values: our colours
        )

    def forward(self, x):
        # 1. Passing our image through the chacateristics extractor
        x_features = self.features(x)
        x_flat = self.flatten(x_features)
        
        # 2. Obtaining final predictions
        out_forma = self.head_forma(x_flat)
        out_color = self.head_color(x_flat)
        
        # 2 Results for each image
        return out_forma, out_color

modelo = ShapeColorCNN()
print(modelo)


# 3. TRAINING CONFIGURATION
# Using GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Moving our model to the selected device
modelo = modelo.to(device)

# Two loss functions
# We need to calculate the error for both heads (Form and Color)
# Using CrossEntropyLoss for multi-class classification
criterion_forma = nn.CrossEntropyLoss()
criterion_color = nn.CrossEntropyLoss()

# The Optimizer (Updates network weights based on the error)
optimizer = optim.Adam(modelo.parameters(), lr=0.001)


# 4. THE TRAINING LOOP
EPOCHS = 10 # How many times the network will see the full dataset

print("\n--- Starting Training ---")
modelo.train() # Set network to "training mode"

for epoch in range(EPOCHS):
    loss_total_acumulada = 0.0
    
    # Extracting batches of 32 from our DataLoader
    for batch_idx, (imagenes, etiquetas_forma, etiquetas_color) in enumerate(dataloader):
        # Move data to the same device as the model (GPU/CPU)
        imagenes = imagenes.to(device)
        etiquetas_forma = etiquetas_forma.to(device)
        etiquetas_color = etiquetas_color.to(device)
        
        # 1. Reset gradients (clear the board from previous iteration)
        optimizer.zero_grad()
        
        # 2. Forward pass: The network tries to guess
        prediccion_forma, prediccion_color = modelo(imagenes)
        
        # 3. Calculate the error (Loss) for each head independently
        loss_forma = criterion_forma(prediccion_forma, etiquetas_forma)
        loss_color = criterion_color(prediccion_color, etiquetas_color)
        
        # 4. Sum the errors: Total error is failing at form + failing at color
        loss_total = loss_forma + loss_color
        
        # 5. Backward pass: Calculate how each weight should change
        loss_total.backward()
        
        # 6. Optimize: Update the mathematical weights
        optimizer.step()
        
        loss_total_acumulada += loss_total.item()
        
    # Print progress at the end of each Epoch
    loss_media = loss_total_acumulada / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Total Error: {loss_media:.4f}")

print("--- Training Finished ---")


# 5. ONNX EXPORT (The bridge to C++)
print("\n--- Exporting model to ONNX ---")

# 1. Set model to "Evaluation" (Inference) mode.
# This locks the weights and disables training-specific layers like Dropout.
modelo.eval()

# 2. Create a "dummy" tensor simulating the C++ camera input
# Dimensions: (Batch of 1, 3 RGB channels, 64 height, 64 width)
dummy_input = torch.randn(1, 3, 64, 64).to(device)

# 3. Final file name
onnx_file_name = "vision_ai_model.onnx"

# 4. Execute the export
torch.onnx.export(
    modelo,                      # The trained model
    dummy_input,                 # The sample image for ONNX to understand the size
    onnx_file_name,              # Output file name
    export_params=True,          # Save the trained mathematical weights
    opset_version=11,            # Standard version, highly compatible with OpenCV C++
    do_constant_folding=True,    # Internal optimization
    input_names=['camera_input'],                 # Name we will use in C++ to feed the image
    output_names=['output_form', 'output_color']  # Names to retrieve results in C++
)

print(f"Success! Your neural network has been packaged and saved as '{onnx_file_name}'.")
