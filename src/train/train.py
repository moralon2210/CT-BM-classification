from ..data.data_processing import prepare_data
from ..data.dataset import create_datasets, create_dataloaders, visualize_random_sample
from ..model import EfficientV2M
import pandas as pd
import torch
import numpy as np
import random
from multiprocessing import freeze_support
from pathlib import Path
from sklearn.metrics import fbeta_score, recall_score, confusion_matrix, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# Import and run training loop
from .train_loop import train_loop
from .train_log import save_history,plot_training_metrics

def main():
    freeze_support()  # Required for Windows multiprocessing
    
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    csv_path = "./Dataset/train/labels1.csv"
    images_folder = "./Dataset/train/CTs"

    # ==================== STEP 1: DATA PROCESSING ====================
    print("\n" + "="*60)
    print("STEP 1: Data processing")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    train_data_dict, val_data_dict, test_data_dict,alpha = prepare_data(df, images_folder)
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data_dict, val_data_dict, test_data_dict
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=8, 
        num_workers=6,
        prefetch_factor=3
    )
    print(f" Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    # Visualize a random sample from training set
    #print("\nVisualizing random training sample...")
    #visualize_random_sample(train_dataset)

    # Init the model and device
    model = EfficientV2M()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Train the model - model checkpoints are saved
    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device=device,  # Auto-detect GPU/CPU
        save_dir="./checkpoints",
        focal_alpha=alpha,  # Based on negetive\positive class frequency
        focal_gamma=2.0,    # Standard focusing parameter
        )

    save_history(history)
    plot_training_metrics(history) 

    # ==================== STEP 3: TESTING ====================
    print("\n" + "="*60)
    print("STEP 3: Testing")
    print("="*60)
    
    # Create results directory
    Path("./results").mkdir(exist_ok=True)
    
    # Load best model
    checkpoint = torch.load("./checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    # Get image names from test dataset
    image_names = [Path(sample['image']).stem for sample in test_dataset.data]
    
    # Test
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing', leave=True)
        for batch in test_pbar:
            images = batch['image'].to(device, dtype=torch.float32)
            labels = batch['label'].cpu().numpy()
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > checkpoint['best_threshold']).astype(int)
            
            all_labels.extend(labels.flatten())
            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    f2 = fbeta_score(all_labels, all_preds, beta=2)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"\nTest F2-Score: {f2:.4f} | Test Recall: {recall:.4f} | Test Precision: {precision:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Visualize and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("./results/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to ./results/confusion_matrix.png")
    
    # Save results
    results = {
        'f2_score': float(f2),
        'recall': float(recall),
        'precision': float(precision),
    }
    
    pd.DataFrame([results]).to_csv("./results/test_results.csv", index=False)
    
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    cm_df.to_csv("./results/confusion_matrix.csv")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'image_name': image_names,
        'true_label': all_labels,
        'probability': all_probs,
        'prediction': all_preds
    })
    predictions_df.to_csv("./results/test_predictions.csv", index=False)
    
    print("\nTesting completed! Results saved to ./results/")
    print("="*60)

if __name__ == '__main__':
    main()
