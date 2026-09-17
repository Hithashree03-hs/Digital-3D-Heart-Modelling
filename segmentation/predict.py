import torch
import matplotlib.pyplot as plt

from dataset import ACDCDataset
from unet import UNet

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model
# -----------------------
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_model.pth", map_location=device))
model.eval()

print("Model Loaded Successfully!")

# -----------------------
# Load Dataset
# -----------------------
dataset = ACDCDataset("dataset/ACDC/test")

# Choose one sample
image, mask = dataset[0]

image = image.unsqueeze(0).to(device)

# -----------------------
# Prediction
# -----------------------
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

image = image.squeeze().cpu().numpy()
mask = mask.numpy()

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image, cmap="gray")
plt.title("MRI")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(prediction)
plt.title("Prediction")
plt.axis("off")

print("Before show")
plt.show()
print("After show")