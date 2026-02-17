import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CopyItemsd, ConcatItemsd, DeleteItemsd,
    Resized, RandFlipd, Rand2DElasticd, EnsureTyped
)
from monai.data import CacheDataset, Dataset, DataLoader
from monai.utils import set_determinism

# 1. Define CT Windows for Brain Metastasis Detection
# Formula: HU_min = WL - WW/2, HU_max = WL + WW/2
# 
# Window 1: The Contrast Booster (WW=35, WL=35) -> [17.5, 52.5]
#   Purpose: Maximizes Signal-to-Noise Ratio for subtle lesion detection
#   What to look for: Loss of Gray-White ribbon, isodense lesions
#
# Window 2: The Baseline/Standard (WW=80, WL=40) -> [0, 80]
#   Purpose: Diagnostic anchor for true brain appearance
#   What to look for: General symmetry, left-right hemisphere comparison
#
# Window 3: The Edema Amplifier (WW=200, WL=40) -> [-60, 140]
#   Purpose: Captures global mass effect and swelling gradients
#   What to look for: Midline shift, sulcal effacement, tumor footprint
#
WINDOWS = [
    (17.5, 52.5),   # Contrast Booster
    (0, 80),         # Baseline (Standard)
    (-60, 140)       # Edema Amplifier
]

# 2. Training Transforms
def transformers(mode=None, seed=42):
    if mode=='train':
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            
            # RESAMPLING: Ensures 1 pixel = 1.0mm physically (The "Real World" spacing)
            Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode="bilinear"),

            # TRIPLE WINDOWING: Stacks 3 HU windows into 3 channels (RGB-like for CNNs)
            CopyItemsd(keys=["image"], times=2, names=["image_w2", "image_w3"]),
            ScaleIntensityRanged(keys=["image"],    a_min=WINDOWS[0][0], a_max=WINDOWS[0][1], b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["image_w2"], a_min=WINDOWS[1][0], a_max=WINDOWS[1][1], b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["image_w3"], a_min=WINDOWS[2][0], a_max=WINDOWS[2][1], b_min=0.0, b_max=1.0, clip=True),
            ConcatItemsd(keys=["image", "image_w2", "image_w3"], name="image", dim=0),
            DeleteItemsd(keys=["image_w2", "image_w3"]),  # Clean up intermediate keys

            Resized(keys=["image"], spatial_size=(224, 224)),

            # AUGMENTATIONS (reduced elastic deformation for brain anatomy preservation)
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            Rand2DElasticd(keys=["image"], spacing=(20, 20), magnitude_range=(0.5, 1.0), prob=0.1),
            
            EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float32, track_meta=False),
        ])
        # Note: MONAI transforms handle their own seeding internally via set_determinism()

    elif mode is None:
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode="bilinear"),
            
            CopyItemsd(keys=["image"], times=2, names=["image_w2", "image_w3"]),
            ScaleIntensityRanged(keys=["image"],    a_min=WINDOWS[0][0], a_max=WINDOWS[0][1], b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["image_w2"], a_min=WINDOWS[1][0], a_max=WINDOWS[1][1], b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["image_w3"], a_min=WINDOWS[2][0], a_max=WINDOWS[2][1], b_min=0.0, b_max=1.0, clip=True),
            ConcatItemsd(keys=["image", "image_w2", "image_w3"], name="image", dim=0),
            DeleteItemsd(keys=["image_w2", "image_w3"]),  # Clean up intermediate keys

            Resized(keys=["image"], spatial_size=(224, 224)),
            EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float32, track_meta=False),
        ])
        # Note: MONAI transforms handle their own seeding internally via set_determinism()

# 3. Create Datasets
def create_datasets(train_data_dict, val_data_dict, test_data_dict, seed=42, cache_rate=1.0):
    """
    Create train, validation, and test datasets with appropriate transforms.

    """
    # Set MONAI determinism for reproducible augmentations
    set_determinism(seed=seed)
    
    # Create transforms
    train_transforms = transformers(mode='train', seed=seed)
    val_test_transforms = transformers(seed=seed)
    
    # Create datasets with caching for faster training
    train_dataset = CacheDataset(data=train_data_dict, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    val_dataset = CacheDataset(data=val_data_dict, transform=val_test_transforms, cache_rate=cache_rate, num_workers=8)
    test_dataset = CacheDataset(data=test_data_dict, transform=val_test_transforms, cache_rate=cache_rate, num_workers=8)
    
    print(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test | Cache rate: {cache_rate}")
    
    return train_dataset, val_dataset, test_dataset

# 4. Create DataLoaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                       batch_size=32, num_workers=4, prefetch_factor=2):
    """
    Create train, validation, and test dataloaders.
        train_loader, val_loader, test_loader
    """
    # Create dataloaders with optimized settings for Windows + CUDA
    # Note: persistent_workers=False is faster on Windows with MONAI+CUDA
    # prefetch_factor: number of batches loaded in advance per worker (default=2)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, persistent_workers=False, pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, persistent_workers=False, pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, persistent_workers=False, pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print(f" Created dataloaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches | Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader

# 5. Visualize Random Sample
def visualize_random_sample(dataset):
    """
    Visualize a random sample from the dataset showing all 3 CT window channels.
    
    Args:
        dataset: MONAI Dataset object
    """
    # Get random sample
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset.__getitem__(idx)
    
    # Extract image and label
    image = sample["image"]  # Shape: (3, 224, 224)
    label = sample["label"].item() if torch.is_tensor(sample["label"]) else sample["label"]
    
    # ============== SANITY CHECKS ==============
    # Convert to numpy for checks if tensor
    if torch.is_tensor(image):
        img_array = image.cpu().numpy()
    else:
        img_array = image
    
    # Check 1: Shape (3 channels, 224x224)
    expected_shape = (3, 224, 224)
    actual_shape = img_array.shape
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"
    
    # Check 2: Number of channels
    num_channels = img_array.shape[0]
    assert num_channels == 3, f"Expected 3 channels, got {num_channels}"
    
    # Check 3: Value range (0-1 for grayscale)
    min_val = img_array.min()
    max_val = img_array.max()
    in_range = (min_val >= 0.0) and (max_val <= 1.0)
    assert in_range, f"Values outside [0, 1] range: [{min_val}, {max_val}]"
    
    print(f"\nSanity checks passed: Shape {actual_shape} | Range [{min_val:.4f}, {max_val:.4f}]")
    
    # Channel names with window settings and purposes
    channel_info = [
        {
            "name": "Contrast Booster (WW=35, WL=35)",
            "purpose": "Max SNR - Subtle lesion detection",
            "look_for": "Gray-White ribbon loss\nIsodense lesions"
        },
        {
            "name": "Baseline/Standard (WW=80, WL=40)",
            "purpose": "Diagnostic anchor",
            "look_for": "General symmetry\nL-R hemisphere comparison"
        },
        {
            "name": "Edema Amplifier (WW=200, WL=40)",
            "purpose": "Global mass effect",
            "look_for": "Midline shift\nSulcal effacement\nTumor footprint"
        }
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Sample #{idx} - Label: {label}", fontsize=16, fontweight='bold')
    
    # Plot each channel
    for i, (ax, info) in enumerate(zip(axes, channel_info)):
        # Convert to numpy and squeeze if needed
        if torch.is_tensor(image):
            channel_img = image[i].cpu().numpy()
        else:
            channel_img = image[i]
        
        ax.imshow(channel_img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{info['name']}\n{info['purpose']}", 
                    fontsize=11, fontweight='bold', pad=10)
        
        # Add text box with what to look for
        ax.text(0.02, 0.98, info['look_for'], 
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig
