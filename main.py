from src.data.data_utils import prepare_data
from src.data.dataset import create_datasets, create_dataloaders, visualize_random_sample
import pandas as pd
import torch
import numpy as np
import random

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

csv_path = "./Dataset/labels1.csv"
images_folder = "./Dataset/CTs"

# Load data
df = pd.read_csv(csv_path)

# Prepare dataset splits
train_data_dict, val_data_dict, test_data_dict = prepare_data(df, images_folder)

# Create datasets
train_dataset, val_dataset, test_dataset = create_datasets(
    train_data_dict, val_data_dict, test_data_dict
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=32, 
    num_workers=4
)

# Visualize a random sample from training set
print("\nVisualizing random training sample...")
visualize_random_sample(train_dataset)
