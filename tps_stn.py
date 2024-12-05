#!/usr/bin/env python3

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.unsqueeze(1) - control_points.unsqueeze(0)
    dist_sq = pairwise_diff.pow(2).sum(-1)
    repr_matrix = dist_sq * torch.log(dist_sq + 1e-6)
    return repr_matrix


class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        N = target_control_points.size(0)
        self.N = N
        self.target_control_points = target_control_points

        # Create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, N] = 1
        forward_kernel[N, :N] = 1
        forward_kernel[:N, N+1:N+3] = target_control_points
        forward_kernel[N+1:N+3, :N] = target_control_points.transpose(0, 1)

        # Compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # Create target coordinate matrix
        HW = target_height * target_width
        target_coordinate = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, steps=target_height),
                torch.linspace(-1, 1, steps=target_width),
                indexing='ij'
            ),
            dim=2
        ).view(-1, 2)

        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr,
            torch.ones(HW, 1),
            target_coordinate
        ], dim=1)

        # Register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)


    def forward(self, source_control_points):
        batch_size = source_control_points.size(0)
        N = source_control_points.size(1)
        Y = torch.cat([
            source_control_points,
            torch.zeros(batch_size, 3, 2, device=source_control_points.device)
        ], dim=1)

        mapping_matrix = torch.matmul(self.inverse_kernel.unsqueeze(0), Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        source_coordinate = source_coordinate.permute(1, 0, 2)
        return source_coordinate


class TPS_STN(nn.Module):
    def __init__(self, img_size, grid_size=5):
        super(TPS_STN, self).__init__()
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_control_points = grid_size * grid_size

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)

        )

        # Regressor for the control points
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_control_points * 2)
        )

        # Initialize the control points
        self.init_control_points()


    def init_control_points(self):
        # Create canonical grid
        control_points = torch.linspace(-1, 1, steps=self.grid_size)
        grid = torch.meshgrid(control_points, control_points, indexing='ij')
        control_points = torch.stack(grid, dim=-1).reshape(-1, 2) 
        self.register_buffer('canonical_control_points', control_points)


    def forward(self, x):
        batch_size = x.size(0)
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        delta_control_points = self.fc(xs)
        delta_control_points = delta_control_points.view(batch_size, self.num_control_points, 2)
        # Add displacement to canonical control points
        control_points = self.canonical_control_points.unsqueeze(0) + delta_control_points
        return control_points
