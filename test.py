#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lpips import LPIPS

from models import Generator
from dataset import SatelliteStreetDataset
from utils import denormalize

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint Loaded: {checkpoint_path}, Epoch {epoch}")
    return model, epoch

def test():
    # Data transformations for input images
    transform_sat = transforms.Compose([
        transforms.Resize((350, 350)),  # Height x Width
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data transformations for output images
    transform_street = transforms.Compose([
        transforms.Resize((112, 616)),  # Height x Width
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test dataset and loader
    test_dataset = SatelliteStreetDataset(
        root_dir='data',
        split='test',
        transform_sat=transform_sat,
        transform_street=transform_street
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model
    generator = Generator(
        img_size=(350, 350),
        output_size=(112, 616),
        patch_size=14,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=512
    ).to(DEVICE)

    # Load model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'generator.pth')
    generator, start_epoch = load_checkpoint(checkpoint_path, generator)
    generator.eval()

    l1_loss_fn = nn.L1Loss().to(DEVICE)
    mse_loss_fn = nn.MSELoss().to(DEVICE)

    # Initialize LPIPS model
    lpips_loss_fn = LPIPS(net='vgg').to(DEVICE)

    # Initialize metrics
    total_l1_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        loop = tqdm(test_loader, leave=True)
        for idx, batch in enumerate(loop):
            sat_images = batch['sat_image'].to(DEVICE)
            real_street_images = batch['street_image'].to(DEVICE)

            # Generate images
            generated_images = generator(sat_images)

            # Compute L1 loss
            loss_L1 = l1_loss_fn(generated_images, real_street_images)
            total_l1_loss += loss_L1.item()

            # Compute LPIPS
            lpips_value = lpips_loss_fn(generated_images, real_street_images)
            total_lpips += lpips_value.item()

            # Denormalize images for SSIM and PSNR computation
            generated_images_denorm = denormalize(generated_images)
            real_street_images_denorm = denormalize(real_street_images)

            # Convert tensors to NumPy arrays
            generated_np = generated_images_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            real_np = real_street_images_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Compute SSIM and PSNR
            #ssim_value = compare_ssim(real_np, generated_np, multichannel=True, data_range=1)
            ssim_value = compare_ssim(real_np, generated_np, channel_axis=2, data_range=1)
            psnr_value = compare_psnr(real_np, generated_np, data_range=1)
            total_ssim += ssim_value
            total_psnr += psnr_value

            # Save images
            sat_images_denorm = denormalize(sat_images)
            input_np = sat_images_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Plot the images side by side
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(input_np)
            axs[0].set_title("Input")
            axs[0].axis('off')

            axs[1].imshow(real_np)
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')

            axs[2].imshow(generated_np)
            axs[2].set_title("Generated")
            axs[2].axis('off')

            # Save the figure
            filename = os.path.join(RESULTS_DIR, f'test_{idx}.png')
            plt.savefig(filename)
            plt.close()

            # Update progress bar
            loop.set_postfix(L1=loss_L1.item(), SSIM=ssim_value, PSNR=psnr_value, LPIPS=lpips_value.item())

        # Compute average metrics
        num_samples = len(test_loader)
        avg_l1_loss = total_l1_loss / num_samples
        avg_ssim = total_ssim / num_samples
        avg_psnr = total_psnr / num_samples
        avg_lpips = total_lpips / num_samples

        print(f"\nAverage L1 Loss on Test Set: {avg_l1_loss}")
        print(f"Average SSIM on Test Set: {avg_ssim}")
        print(f"Average PSNR on Test Set: {avg_psnr}")
        print(f"Average LPIPS on Test Set: {avg_lpips}")


if __name__ == '__main__':
    test()
