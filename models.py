#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from tps_stn import TPSGridGen, TPS_STN


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.25):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        # attn_output = self.dropout1(attn_output)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        # ff_output = self.dropout2(ff_output)
        x = self.layernorm2(x + ff_output)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),        

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.Dropout(0.5)         
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, img_size=(350, 350), output_size=(112, 616), patch_size=14, embed_dim=256, num_layers=4, num_heads=8, ff_dim=512, num_control_points=20):
        super(Generator, self).__init__()
        self.img_size = img_size  
        self.output_size = output_size  
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # TPS-STN module
        #self.stn = TPS_STN(img_size=img_size, grid_size=5)
        # TPS grid generator
        #target_control_points = self.stn.canonical_control_points
        #self.tps = TPSGridGen(img_size[0], img_size[1], target_control_points)

        # Calculate the number of patches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
            for _ in range(num_layers)
        ])

        # Decoder (same as before)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        # Final layers
        self.final_layers = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )


    def forward(self, x):
        batch_size = x.size(0)
        
        # Get control points from STN
        #source_control_points = self.stn(x)
        
        # Generate TPS grid
        #source_coordinate = self.tps(source_control_points)
        
        # Reshape source_coordinate
        #grid = source_coordinate.reshape(batch_size, self.img_size[0], self.img_size[1], 2)

        # Apply TPS transformation
        #x = F.grid_sample(x, grid, align_corners=True)

        # Patch embedding
        x = self.patch_embed(x)
        H_prime, W_prime = x.shape[2], x.shape[3]
        x = x.flatten(2).permute(2, 0, 1)

        # Add positional encoding
        x = x + self.pos_embed.permute(1, 0, 2)

        # Transformer encoding
        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.permute(1, 2, 0).view(batch_size, self.embed_dim, H_prime, W_prime)

        # Decode
        x = self.decoder(x)

        # Upsample to the desired output size
        x = nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)

        x = self.final_layers(x)

        return x #, source_control_points


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input is (input_channels) x 112 x 616
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Output size depends on input size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Output is a patch map
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
