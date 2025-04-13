import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DenseHazeNPYDataset(Dataset):
    def __init__(self, hazy_path="hazy.npy", gt_path="GT.npy", transform=None):
        """
        Dataset for loading pre-saved grayscale .npy arrays.
        Each image is expected to have shape (256, 256).
        """
        self.hazy_data = np.load(hazy_path, allow_pickle=True)
        self.gt_data = np.load(gt_path, allow_pickle=True)
        assert len(self.hazy_data) == len(self.gt_data), "Hazy and GT data length mismatch."

        self.transform = transform

    def __len__(self):
        return len(self.hazy_data)

    def __getitem__(self, idx):
        hazy = self.hazy_data[idx]
        gt = self.gt_data[idx]

        # Convert to grayscale if needed
        if len(hazy.shape) == 3 and hazy.shape[2] == 3:
            hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2GRAY)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1] and add channel dimension
        hazy = torch.tensor(hazy / 255.0, dtype=torch.float32).unsqueeze(0)
        gt = torch.tensor(gt / 255.0, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            hazy = self.transform(hazy)
            gt = self.transform(gt)

        return gt, hazy  # (ground truth, hazy input)


def build_npy_from_images(input_dir, output_name="GT.npy", img_size=256):
    """
    Converts a folder of images into a .npy file.
    All images are resized to (img_size, img_size).
    """
    image_list = []
    files = sorted(os.listdir(input_dir))

    print(f"🔄 Processing images from: {input_dir}")
    for filename in tqdm(files):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            image_list.append(img)

    arr = np.array(image_list)
    np.save(output_name, arr)
    print(f"✅ Saved {len(image_list)} images to {output_name}")


# Optional usage example
if __name__ == "__main__":
    # Generate .npy files from GT and hazy folders
    build_npy_from_images("GT", "GT.npy")
    build_npy_from_images("hazy", "hazy.npy")

    # Load and test the dataset
    dataset = DenseHazeNPYDataset("hazy.npy", "GT.npy")
    print(f"Loaded {len(dataset)} samples.")
    sample = dataset[0]
    print("Sample shape (GT, Hazy):", sample[0].shape, sample[1].shape)
