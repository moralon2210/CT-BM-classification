import torch
import pandas as pd
from pathlib import Path
from monai.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import EfficientV2M
from src.data.dataset import transformers

def load_best_model(checkpoint_path, device):
    """Load the best model from checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = EfficientV2M()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation F-beta: {checkpoint['val_fbeta']:.4f}")
    print(f"Best Threshold: {checkpoint['best_threshold']:.4f}")
    
    return model, checkpoint

def prepare_inference_data(images_folder):
    """Prepare data dictionary for inference from a folder of images"""
    images_folder = Path(images_folder)
    image_files = list(images_folder.glob("*.dcm"))
    
    if not image_files:
        raise ValueError(f"No .dcm files found in {images_folder}")
    
    print(f"Found {len(image_files)} images for inference")
    
    data_dict = [{"image": str(img_path)} for img_path in sorted(image_files)]
    image_names = [img_path.name for img_path in sorted(image_files)]
    
    return data_dict, image_names

def run_inference(model, dataloader, device, threshold):
    """Run inference on a dataloader"""
    predictions = []
    probabilities = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device, dtype=torch.float32)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            probabilities.extend(probs.flatten().tolist())
            predictions.extend(preds.flatten().tolist())
    
    return predictions, probabilities

def save_predictions(image_names, predictions, probabilities, output_path):
    """Save predictions to a CSV file"""
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'image_name': image_names,
        'prediction': predictions,
        'probability': probabilities
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    positive_count = sum(predictions)
    total_count = len(predictions)
    print(f"\nSummary:")
    print(f"Total images: {total_count}")
    print(f"Positive (Brain Metastasis): {positive_count} ({positive_count/total_count*100:.1f}%)")
    print(f"Negative (No Metastasis): {total_count - positive_count} ({(total_count-positive_count)/total_count*100:.1f}%)")
    
    return results_df

def main(
    images_folder="./Dataset/inference/CTs",
    checkpoint_path="./inference_model/best_model.pth",
    output_path="./results/inference/predictions.csv",
    batch_size=8,
    num_workers=4
):
    """Main inference pipeline"""
    print("="*60)
    print("Brain Metastasis Detection - Inference Pipeline")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model and get threshold
    model, checkpoint = load_best_model(checkpoint_path, device)
    threshold = checkpoint['best_threshold']
    
    # Prepare data
    data_dict, image_names = prepare_inference_data(images_folder)
    
    # Create dataloader
    inference_dataset = Dataset(data=data_dict, transform=transformers(mode='inference'))
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Run inference and save
    predictions, probabilities = run_inference(model, inference_loader, device, threshold)
    results_df = save_predictions(image_names, predictions, probabilities, output_path)
    
    print("Inference completed successfully!")
    print("="*60)
    
    return results_df

if __name__ == "__main__":
    results = main()
