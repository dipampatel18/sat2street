#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    filename = os.path.join(checkpoint_dir, f'{model_name}_{epoch}.pth')
    # filename = os.path.join(checkpoint_dir, f'{model_name}.pth')
    torch.save(checkpoint, filename)
    print(f"Checkpoint Saved: {filename}!")

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint Loaded: {checkpoint_path}, Epoch {epoch}")
    return epoch

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor(std).view(1, 3, 1, 1).to(DEVICE)
    image = image * std + mean
    return torch.clamp(image, 0, 1)

def save_examples(gen, val_loader, epoch, results_dir, mode, num_examples=3):
    gen.eval()
    with torch.no_grad():
        # Get the validation dataset
        val_dataset = val_loader.dataset
        n_samples = len(val_dataset)

        # Randomly select num_examples indices
        indices = np.random.choice(n_samples, num_examples, replace=False)

        # Ensure the 'results' directory exists
        os.makedirs(results_dir, exist_ok=True)

        for i, idx in enumerate(indices):
            sample = val_dataset[idx]
            x = sample['sat_image'].unsqueeze(0).to(DEVICE)
            y = sample['street_image'].unsqueeze(0).to(DEVICE)
            y_fake, _ = gen(x)

            # Denormalize images
            x_denorm = denormalize(x)
            y_denorm = denormalize(y)
            y_fake_denorm = denormalize(y_fake)

            # Convert tensors to NumPy arrays
            input_np = x_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            target_np = y_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = y_fake_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Plot the images side by side
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(input_np)
            axs[0].set_title("Input")
            axs[0].axis('off')

            axs[1].imshow(target_np)
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')

            axs[2].imshow(output_np)
            axs[2].set_title("Generated")
            axs[2].axis('off')

            # Save the figure
            filename = os.path.join(results_dir, f'{mode}/{mode}_{epoch}_{i}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved Example: {filename}")
    gen.train()

def save_transformed_examples(generator, val_loader, epoch, results_dir, mode, num_examples=3):
    generator.eval()
    with torch.no_grad():
        val_dataset = val_loader.dataset
        n_samples = len(val_dataset)
        indices = np.random.choice(n_samples, num_examples, replace=False)

        for i, idx in enumerate(indices):
            sample = val_dataset[idx]
            x = sample['sat_image'].unsqueeze(0).to(DEVICE)
            batch_size = x.size(0)

            # Get control points from STN
            source_control_points = generator.stn(x)

            # Generate TPS grid
            source_coordinate = generator.tps(source_control_points)
            # Reshape source_coordinate to grid
            grid = source_coordinate.reshape(batch_size, generator.img_size[0], generator.img_size[1], 2)

            # Apply TPS transformation
            transformed_x = F.grid_sample(x, grid, align_corners=True)

            # Denormalize and save the transformed images
            transformed_x_denorm = denormalize(transformed_x)
            transformed_np = transformed_x_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Save the image
            filename = os.path.join(results_dir, f'{mode}/{mode}_{epoch}_{i}.png')
            plt.imsave(filename, transformed_np)
            print(f"Saved Transformed Example: {filename}")
    generator.train()
