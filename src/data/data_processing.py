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
        print("No duplicate IDs found. All rows are unique.")
    else:
        duplicate_ids = duplicates['ID'].unique()
        print(f"Found {len(duplicate_ids)} duplicate IDs:")
        for dup_id in duplicate_ids:
            count = len(df[df['ID'] == dup_id])
            print(f"  - {dup_id}: appears {count} times")
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=['ID'], keep='first')
        print(f"\nRemoved {initial_count - len(df)} duplicate rows.")
    
    print(f"Final dataset: {len(df)} unique rows")
    
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
    
    print(f"Found {len(available_ids)} DICOM files in {images_folder}")
    
    # Get initial row count
    initial_count = len(df)
    
    # Find IDs in CSV that don't have corresponding images (vectorized)
    missing_mask = ~df['ID'].isin(available_ids)
    missing_images = df.loc[missing_mask, 'ID'].tolist()
    
    if len(missing_images) == 0:
        print("All IDs in CSV have corresponding image files.")
    else:
        print(f"Found {len(missing_images)} IDs without corresponding images")
        
        # Remove rows with missing images
        df = df[df['ID'].isin(available_ids)]
        print(f"Removed {initial_count - len(df)} rows with missing images.")
    
    print(f"Final dataset: {len(df)} rows with valid images")
    
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
    total_samples = len(y_train_array)
    
    alpha = num_negatives / total_samples
    
    print(f"Negative samples: {num_negatives}")
    print(f"Total samples: {total_samples}")
    print(f"Negative ratio (alpha): {alpha:.4f}")
    
    return alpha


def data_to_dict(x,y, images_folder):
    """
    Prepares the data structure for monai dataset.
        
    Returns:
        list: List of dictionaries with format [{"image": "path/to/img.dcm", "label": 0}, ...]
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
    
    print(f"Created data structure with {len(data_dicts)} samples")
    
    return data_dicts



def prepare_data(df, images_folder):
    """
    Main orchestration function for data preparation.
    """
    print("=" * 60)
    print("Starting data preperation pipeline")
    
        # Step 1: Check and remove duplicate IDs
    print("\nSTEP 1: Checking for duplicate IDs")
    df_clean = check_and_remove_duplicate_ids(df)
    
    # Step 2: Check IDs against image files
    print("\nSTEP 2: Checking IDs against image files")
    clean_df = check_ids_against_images(df_clean, images_folder)
    
    # Optionally save the cleaned CSV
    output_path = "./Dataset/labels_clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")
    
    # Step 3: Prepare data structure
    print("\nSTEP 3: Split to train, validation and test sets (70%, 15%, 15%)")
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(clean_df)

    #calcualte alpha for focal loss
    print("\nSTEP 4: Caclaute alpha for focal loss based on train data imbalance")
    alpha = calculate_alpha(y_train)
    
    # Step 5: Prepare data structure
    print("\nSTEP 5: Preparing data structure for train,val and test")
    train_data_dict = data_to_dict(x_train,y_train, images_folder)
    val_data_dict = data_to_dict(x_val,y_val, images_folder)
    test_data_dict = data_to_dict(x_test,y_test, images_folder)
    
    print("Data preperation completed")
    print("=" * 60)
    
    return train_data_dict,val_data_dict,test_data_dict,alpha
