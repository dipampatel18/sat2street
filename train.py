#!/usr/bin/env python3

import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator
from dataset import SatelliteStreetDataset
from losses import PerceptualLoss, control_points_regularization_loss
from utils import save_checkpoint, load_checkpoint, save_examples, save_transformed_examples

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 1000
BATCH_SIZE = 64
G_LR = 1e-4
D_LR = 1e-5
LAMBDA_ADV = 0.5 #1.0
LAMBDA_L1 = 200.0 #100.0
LAMBDA_PERCEPTUAL = 10.0
LAMBDA_CONTROL_POINTS = 0.1
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'
LOG_DIR = './logs'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def train():
    # Initialize TensorBoard writer
    writer = SummaryWriter(LOG_DIR)

    # Data transformations for input images (satellite images)
    transform_sat = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data transformations for output images (street view images)
    transform_street = transforms.Compose([
        transforms.Resize((112, 616)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Training dataset and loader
    train_dataset = SatelliteStreetDataset(
        root_dir='data',
        split='train',
        transform_sat=transform_sat,
        transform_street=transform_street
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Validation dataset and loader
    val_dataset = SatelliteStreetDataset(
        root_dir='data',
        split='val',
        transform_sat=transform_sat,
        transform_street=transform_street
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize models
    generator = Generator(
        img_size=(350, 350),
        output_size=(112, 616),
        patch_size=14,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=512
    ).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Initialize optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=G_LR, betas=(0.5, 0.999)) #, weight_decay=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999)) #, weight_decay=1e-4) 

    # # Initialize learning rate schedulers
    # scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)       
    # scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)       

    # Initialize scaler for mixed precision training
    scaler_G = torch.amp.GradScaler("cuda")                                                                                  
    scaler_D = torch.amp.GradScaler("cuda")                                                                                  

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss().to(DEVICE)
    l1_loss = nn.L1Loss().to(DEVICE)
    # perceptual_loss = PerceptualLoss().to(DEVICE)                                                                            

    # Training loop
    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()
        loop = tqdm(train_loader, leave=True)
        running_loss_G = 0.0
        running_loss_D = 0.0
        val_loss = 0.0

        for idx, batch in enumerate(loop):
            sat_images = batch['sat_image'].to(DEVICE)
            real_street_images = batch['street_image'].to(DEVICE)
            batch_size = sat_images.size(0)

            ###### Train Generator ######
            optimizer_G.zero_grad()

            with torch.amp.autocast('cuda'):                
                # Generate images
                #generated_images, source_control_points = generator(sat_images)
                generated_images = generator(sat_images)

                # Adversarial loss
                pred_fake = discriminator(generated_images)
                valid = torch.ones_like(pred_fake, device=DEVICE)
                loss_G_adv = adversarial_loss(pred_fake, valid)

                # L1 loss
                loss_G_L1 = l1_loss(generated_images, real_street_images) * LAMBDA_L1

                # Perceptual loss
                # loss_G_perceptual = perceptual_loss(generated_images, real_street_images) * LAMBDA_PERCEPTUAL   

                #control_points_loss = control_points_regularization_loss(source_control_points, generator.stn.canonical_control_points.unsqueeze(0).expand(batch_size, -1, -1)) * LAMBDA_CONTROL_POINTS

                # Total generator loss
                loss_G = loss_G_adv + loss_G_L1 #+ control_points_loss#+ loss_G_perceptual                                             

            # # Backward and optimize
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            # Backward and optimize using gradient scaler
            #scaler_G.scale(loss_G).backward()
            #scaler_G.step(optimizer_G)
            #scaler_G.update()

            if idx % 2 == 0:
                ###### Train Discriminator ######
                optimizer_D.zero_grad()

                with torch.amp.autocast('cuda'):
                    # Discriminator output for real images
                    noise = torch.randn_like(real_street_images) * 0.05
                    real_street_images = real_street_images + noise
                    pred_real = discriminator(real_street_images)
                    valid = torch.ones_like(pred_real, device=DEVICE) * 0.9
                    loss_D_real = adversarial_loss(pred_real, valid)

                    # Discriminator output for fake images
                    pred_fake = discriminator(generated_images.detach())
                    fake = torch.zeros_like(pred_fake, device=DEVICE) * 0.1
                    loss_D_fake = adversarial_loss(pred_fake, fake)

                    # Total discriminator loss
                    loss_D = (loss_D_real + loss_D_fake) * 0.5

                # # Backward and optimize
                loss_D.backward()
                optimizer_D.step()

                # Backward and optimize using gradient scaler
                #scaler_D.scale(loss_D).backward()
                #scaler_D.step(optimizer_D)
                #scaler_D.update()

            # Update tqdm loop description
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(
                loss_G=loss_G.item(),
                loss_D=loss_D.item(),
                loss_G_adv=loss_G_adv.item(),
                loss_G_L1=loss_G_L1.item()
            )

            # Accumulate losses for logging
            running_loss_G += loss_G.item()
            if idx % 2 == 0:                                                                   
                running_loss_D += loss_D.item()

            # Log training losses every 100 iterations
            if idx % 100 == 0:
                global_step = epoch * len(train_loader) + idx
                writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
                if idx % 2 == 0:                                                               
                    writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)

        # Compute average losses
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / (len(train_loader) // 2)                                   

        # Log average losses for the epoch
        writer.add_scalar('Loss/Generator_epoch', avg_loss_G, epoch)
        writer.add_scalar('Loss/Discriminator_epoch', avg_loss_D, epoch)

        if epoch % 5 == 0:
            save_examples(generator, train_loader, epoch, RESULTS_DIR, mode='train', num_examples=3)
            # Validation loop
            generator.eval()
            with torch.no_grad():
                val_loss = 0.0
                for idx, batch in enumerate(val_loader):
                    sat_images = batch['sat_image'].to(DEVICE)
                    real_street_images = batch['street_image'].to(DEVICE)

                    with torch.amp.autocast('cuda'):
                        # Generate images
                        #generated_images, source_control_points = generator(sat_images)
                        generated_images = generator(sat_images)

                        # L1 loss
                        loss_L1 = l1_loss(generated_images, real_street_images)
                        val_loss += loss_L1.item()

                val_loss /= len(val_loader)
                print(f"Validation L1 Loss: {val_loss}")

                # Log validation loss
                writer.add_scalar('Loss/Validation_L1', val_loss, epoch)

                # Save some examples from validation set
                save_examples(generator, val_loader, epoch, RESULTS_DIR, mode='val', num_examples=3)
                save_transformed_examples(generator, val_loader, epoch, RESULTS_DIR, mode='transformed', num_examples=3)

                if epoch % 10 == 0:
                   # Save model checkpoints
                   save_checkpoint(generator, optimizer_G, epoch, CHECKPOINT_DIR, 'generator')
                   save_checkpoint(discriminator, optimizer_D, epoch, CHECKPOINT_DIR, 'discriminator')
        
        # # Update learning rates
        # scheduler_G.step(val_loss)
        # scheduler_D.step(val_loss)
                
    writer.close()

if __name__ == '__main__':
    train()
