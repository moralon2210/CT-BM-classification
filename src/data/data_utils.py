import pandas as pd
import os
from pathlib import Path


def check_and_remove_duplicate_ids(csv_path):
    """
    Reads a CSV file and removes duplicate IDs, keeping only unique rows.
    """

    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get initial row count
    initial_count = len(df)
    
    # Check for duplicates in the ID column
    duplicates = df[df.duplicated(subset=['ID'], keep=False)]
    
    if len(duplicates) == 0:
        print("✓ No duplicate IDs found. All rows are unique.")
    else:
        duplicate_ids = duplicates['ID'].unique()
        print(f"⚠ Found {len(duplicate_ids)} duplicate IDs:")
        for dup_id in duplicate_ids:
            count = len(df[df['ID'] == dup_id])
            print(f"  - {dup_id}: appears {count} times")
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=['ID'], keep='first')
        print(f"\n✓ Removed {initial_count - len(df)} duplicate rows.")
    
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
    
    # Find IDs in CSV that don't have corresponding images
    missing_images = []
    for idx, row in df.iterrows():
        if row['ID'] not in available_ids:
            missing_images.append(row['ID'])
    
    if len(missing_images) == 0:
        print(" All IDs in CSV have corresponding image files.")
    else:
        print(f"\nFound {len(missing_images)} IDs without corresponding images")
        
        # Remove rows with missing images
        df = df[df['ID'].isin(available_ids)]
        print(f"\n✓ Removed {initial_count - len(df)} rows with missing images.")
    
    print(f"Final dataset: {len(df)} rows with valid images")
    
    return df


def data_checks(csv_path,images_folder):
    
    # Step 1: Check and remove duplicate IDs
    print("=" * 60)
    print("STEP 1: Checking for duplicate IDs")
    print("=" * 60)
    df_clean = check_and_remove_duplicate_ids(csv_path)
    
    # Step 2: Check IDs against image files
    print("=" * 60)
    print("STEP 2: Checking IDs against image files")
    print("=" * 60)
    clean_df = check_ids_against_images(df_clean, images_folder)
    
    # Optionally save the cleaned CSV
    output_path = "./Dataset/labels_clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned CSV saved to: {output_path}")

    return clean_df
