from src.data.data_processing import prepare_data
from src.data.dataset import create_datasets, create_dataloaders, visualize_random_sample
from src.model import EfficientV2M
import pandas as pd
import torch
import numpy as np
import random
from multiprocessing import freeze_support
# Import and run training loop
from src.train_loop import train_loop

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    
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
    # to test pipeline
    df = df.iloc[:400]

    # Prepare dataset splits
    train_data_dict, val_data_dict, test_data_dict,alpha = prepare_data(df, images_folder)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data_dict, val_data_dict, test_data_dict
    )

    # Create dataloaders
    # Increased num_workers to 6-8 for better GPU utilization
    # prefetch_factor=3 keeps more batches ready in advance
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=4, 
        num_workers=6,  # Increased from 4 to keep GPU fed
        prefetch_factor=3  # Prefetch more batches
    )

    # Visualize a random sample from training set
    #print("\nVisualizing random training sample...")
    #visualize_random_sample(train_dataset)

    # Init the model
    model = EfficientV2M()


    # Train the model - model checkpoints are saved
    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        learning_rate=1e-4,
        device=None,  # Auto-detect GPU/CPU
        save_dir="./checkpoints",
        focal_alpha=alpha,  # Based on negetive\positive class frequency
        focal_gamma=2.0    # Standard focusing parameter
    )
