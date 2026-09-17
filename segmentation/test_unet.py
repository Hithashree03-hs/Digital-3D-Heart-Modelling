import torch
from torch.utils.data import DataLoader

from dataset import HVSMR2DDataset
from unet import UNet


print("\n====================================")
print("HVSMR Training Pipeline Test")
print("====================================")


# -------------------------------------------------
# Dataset
# -------------------------------------------------

dataset = HVSMR2DDataset(
    "../dataset/HVSMR/cropped/cropped"
)

print("\nDataset slices:", len(dataset))


# -------------------------------------------------
# DataLoader
# -------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

images, masks = next(iter(loader))

print("\nBatch information")
print("-----------------")

print("Images:", images.shape)
print("Masks :", masks.shape)

print("Image dtype:", images.dtype)
print("Mask dtype :", masks.dtype)

print("Mask labels:", torch.unique(masks))


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)


# -------------------------------------------------
# Model
# -------------------------------------------------

model = UNet(
    in_channels=1,
    out_channels=9
).to(device)

images = images.to(device)
masks = masks.to(device)


# -------------------------------------------------
# Forward pass
# -------------------------------------------------

print("\nRunning forward pass...")

outputs = model(images)

print("Output shape:", outputs.shape)


# -------------------------------------------------
# Cross Entropy
# -------------------------------------------------

cross_entropy = torch.nn.CrossEntropyLoss()

ce_loss = cross_entropy(
    outputs,
    masks
)

print(
    "Cross-Entropy Loss:",
    ce_loss.item()
)


# -------------------------------------------------
# Dice Loss
# -------------------------------------------------

probabilities = torch.softmax(
    outputs,
    dim=1
)

targets_one_hot = torch.nn.functional.one_hot(
    masks,
    num_classes=9
)

targets_one_hot = targets_one_hot.permute(
    0,
    3,
    1,
    2
).float()

smooth = 1e-6

intersection = (
    probabilities * targets_one_hot
).sum(dim=(0, 2, 3))

denominator = (
    probabilities.sum(dim=(0, 2, 3))
    +
    targets_one_hot.sum(dim=(0, 2, 3))
)

dice = (
    (2.0 * intersection + smooth)
    /
    (denominator + smooth)
)

dice_loss = 1.0 - dice.mean()

print(
    "Dice Loss:",
    dice_loss.item()
)


# -------------------------------------------------
# Combined Loss
# -------------------------------------------------

loss = (
    ce_loss + dice_loss
) / 2.0

print(
    "Combined Loss:",
    loss.item()
)


# -------------------------------------------------
# Backpropagation
# -------------------------------------------------

print("\nTesting backpropagation...")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

optimizer.zero_grad()

loss.backward()

optimizer.step()


print("Backpropagation: SUCCESS")
print("Optimizer step: SUCCESS")


print("\n====================================")
print("Training pipeline test PASSED")
print("====================================")