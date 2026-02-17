import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from pathlib import Path


def check_and_remove_duplicate_ids(df):
    """
    Reads a CSV file and removes duplicate IDs, keeping only unique rows.
    Crucial to prevent data leakage later on in the splits
    """
 
    # Get initial row count
    initial_count = len(df)
    
    # Check for duplicates in the ID column
    duplicates = df[df.duplicated(subset=['ID'], keep=False)]
    
    if len(duplicates) == 0:
        print("No duplicate IDs found")
    else:
        duplicate_ids = duplicates['ID'].unique()
        df = df.drop_duplicates(subset=['ID'], keep='first')
        print(f"Removed {initial_count - len(df)} duplicate rows ({len(duplicate_ids)} duplicate IDs)")
    
    return df


def check_ids_against_images(df, images_folder):
    """
    Checks that every ID in the DataFrame exists in the images folder.
    Removes IDs that don't have corresponding image files.
    
    """
    images_folder = Path(images_folder)
    
    # Get all image files in the folder
    image_files = list(images_folder.glob('*.dcm'))
    
    # Extract IDs from filenames (remove .dcm extension)
    available_ids = set([img.stem for img in image_files])
    
    # Get initial row count
    initial_count = len(df)
    
    # Find IDs in CSV that don't have corresponding images (vectorized)
    missing_mask = ~df['ID'].isin(available_ids)
    missing_images = df.loc[missing_mask, 'ID'].tolist()
    
    if len(missing_images) == 0:
        print(f"All {len(df)} IDs matched with DICOM files")
    else:
        df = df[df['ID'].isin(available_ids)]
        print(f"Matched {len(df)} IDs with DICOM files (removed {initial_count - len(df)} missing)")
    
    return df

def split_train_val_test(df):
    x_train, X_temp, y_train, y_temp = train_test_split(
    df['ID'].tolist(), df['Label'].tolist(), test_size=0.3, stratify=df['Label'], random_state=42)

    x_val, x_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test

def calculate_alpha(y_train):
    """
    Calculates the ratio of negative samples to total samples.
    Used for calculating the alpha parameter in loss functions (e.g., focal loss).
    
    """
    y_train_array = np.array(y_train)
    num_negatives = np.sum(y_train_array == 0)
    num_positives = np.sum(y_train_array == 1)
    total_samples = len(y_train_array)
    
    alpha = np.round(num_negatives / total_samples,4)
    
    print(f"Class balance: {num_negatives} negative, {num_positives} positive | Alpha: {alpha:.4f}")
    
    return alpha


def data_to_dict(x,y, images_folder):
    """
    Prepares the data structure for monai dataset.
        
    """
    images_folder = Path(images_folder)
    data_dicts = []
    
    for idx in range(len(x)):
        image_path = images_folder / f"{x[idx]}.dcm"
        data_dict = {
            "image": str(image_path),
            "label": int(y[idx])
        }
        data_dicts.append(data_dict)
    
    return data_dicts



def prepare_data(df, images_folder):
    """
    Main orchestration function for data preparation.
    """
    # Check and remove duplicate IDs
    df_clean = check_and_remove_duplicate_ids(df)
    
    # Check IDs against image files
    clean_df = check_ids_against_images(df_clean, images_folder)
    
    # Save the cleaned CSV
    output_path = "./Dataset/train/labels_clean.csv"
    clean_df.to_csv(output_path, index=False)
    
    # Split to train, validation and test sets (70%, 15%, 15%)
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(clean_df)
    print(f"Split data: {len(x_train)} train, {len(x_val)} val, {len(x_test)} test")

    # Calculate alpha for focal loss
    alpha = calculate_alpha(y_train)
    
    # Prepare data structure
    train_data_dict = data_to_dict(x_train,y_train, images_folder)
    val_data_dict = data_to_dict(x_val,y_val, images_folder)
    test_data_dict = data_to_dict(x_test,y_test, images_folder)
    
    return train_data_dict,val_data_dict,test_data_dict,alpha
