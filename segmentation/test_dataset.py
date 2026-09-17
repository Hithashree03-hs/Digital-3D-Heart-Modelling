from dataset import HVSMR2DDataset
import torch


print("\nLoading dataset...\n")

dataset = HVSMR2DDataset(
    "../dataset/HVSMR/cropped/cropped"
)

print("\nDataset verification")
print("====================")

print("Total slices:", len(dataset))


# Test a few slices only
test_indices = [
    0,
    100,
    500,
    1000,
    3000,
    5000,
    8000
]


for index in test_indices:

    image, mask = dataset[index]

    print(
        f"\nSlice {index}:"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Mask dtype:",
        mask.dtype
    )

    print(
        "Labels:",
        torch.unique(mask).tolist()
    )


print("\nDataset test completed successfully.")