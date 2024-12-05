# Sat2Street - Dipam Patel

# Install the Packages in conda or virtualenv

pip3 install -r requirements.txt

# Test the Model

`python3 test.py`

This runs inference on the provided test dataset with the trained models (at epoch 350) available in checkpoints dir. This produces plots of Input satellite image, Ground truth street-view image and Generated street-view image.

# Train the Model

- Get access to CVUSA dataset - https://mvrl.cse.wustl.edu/datasets/cvusa/

Place it under `data/train` with `satellite` and `street` folders respectively.

`python3 train.py`

# Note

There is commented code for Spatial Transformer Network (STN) with Thin Plate Spline (TPS) which is still experimental and needs tuning as it performed poorly in comparison to just hybrid approach.
