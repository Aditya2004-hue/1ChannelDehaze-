import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Encoder, Decoder
from dataset import DenseHazeNPYDataset
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# -------------------------
# Settings
# -------------------------
MODEL_PATH = "dehaze_autoencoder.pth"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model
# -------------------------
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)

checkpoint = torch.load(MODEL_PATH)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])
encoder.eval()
decoder.eval()

# -------------------------
# Load dataset
# -------------------------
dataset = DenseHazeNPYDataset("hazy.npy", "GT.npy")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

total_psnr = 0.0
total_ssim = 0.0

print("🔍 Evaluating model...")
with torch.no_grad():
    for idx, (gt, hazy) in enumerate(tqdm(loader)):
        gt = gt.to(DEVICE)
        hazy = hazy.to(DEVICE)

        encoded = encoder(hazy)
        output = decoder(encoded).clamp(0, 1)

        # For metrics
        gt_np = gt.squeeze().cpu().numpy()
        out_np = output.squeeze().cpu().numpy()

        psnr_val = psnr(gt_np, out_np, data_range=1)
        ssim_val = ssim(gt_np, out_np, data_range=1)

        total_psnr += psnr_val
        total_ssim += ssim_val

        # Save side-by-side image
        comparison = torch.cat([hazy.cpu(), output.cpu(), gt.cpu()], dim=3)  # [B, 1, H, W * 3]
        save_image(comparison, os.path.join(SAVE_DIR, f"{idx:02d}_compare.png"))

avg_psnr = total_psnr / len(loader)
avg_ssim = total_ssim / len(loader)

print("\n✅ Evaluation Complete")
print(f"📊 Average PSNR: {avg_psnr:.2f}")
print(f"📊 Average SSIM: {avg_ssim:.4f}")
print(f"🖼️ Comparison images saved in: {SAVE_DIR}/")
