#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        model = models.vgg19(weights='VGG19_Weights.DEFAULT')
        # model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.layers = layers if layers is not None else ['0', '5', '10', '19', '28']
        self.model = model
        self.criterion = nn.L1Loss()


    def forward(self, input, target):
        loss = 0.0
        x = input
        y = target
        for name, layer in self.model._modules.items():
            x = layer(x)
            y = layer(y)
            if name in self.layers:
                loss += self.criterion(x, y)
        return loss


def control_points_regularization_loss(predicted_control_points, canonical_control_points):
    loss = nn.MSELoss()
    return loss(predicted_control_points, canonical_control_points)
