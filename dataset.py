#!/usr/bin/env python3

import os

from PIL import Image
from torch.utils.data import Dataset


class SatelliteStreetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform_sat=None, transform_street=None):
        self.root_dir = root_dir
        self.split = split
        self.transform_sat = transform_sat
        self.transform_street = transform_street

        # Paths to satellite and street view images
        self.sat_dir = os.path.join(root_dir, split, 'satellite')
        self.street_dir = os.path.join(root_dir, split, 'street')

        # List of image file names
        self.sat_images = sorted(os.listdir(self.sat_dir))
        self.street_images = sorted(os.listdir(self.street_dir))

        assert len(self.sat_images) == len(self.street_images), "Mismatch between satellite and street images."


    def __len__(self):
        return len(self.sat_images)


    def __getitem__(self, idx):
        sat_path = os.path.join(self.sat_dir, self.sat_images[idx])
        street_path = os.path.join(self.street_dir, self.street_images[idx])

        # Load images
        sat_image = Image.open(sat_path).convert('RGB')
        street_image = Image.open(street_path).convert('RGB')

        # Apply transformations
        if self.transform_sat:
            sat_image = self.transform_sat(sat_image)
        if self.transform_street:
            street_image = self.transform_street(street_image)

        return {
            'sat_image': sat_image,
            'street_image': street_image
        }
