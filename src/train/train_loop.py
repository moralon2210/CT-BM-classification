import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import fbeta_score, recall_score, auc, precision_recall_curve
import torch.nn.functional as F
import json
import shutil
from .train_utils import binary_focal_loss, find_best_threshold, calculate_metrics

def train_loop(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, 
               device=None, save_dir="./checkpoints", focal_alpha=0.75, focal_gamma=2.0, 
               fbeta_beta=2.0):
    """
    Training loop for the model with Focal Loss for handling class imbalance

    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-6,patience=2)
    
    # Clear and create checkpoint directory
    save_path = Path(save_dir)
    if save_path.exists():
        shutil.rmtree(save_path)
        print(f"Cleared existing checkpoints in {save_dir}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_fbeta': [],
        'val_fbeta': [],
        'train_recall': [],
        'val_recall': [],
        'train_pr_auc': [],
        'val_pr_auc': [],
        'best_thresholds': []
    }
    
    best_val_loss = float('inf')
    best_val_fbeta = 0.0  # Track best F-beta score
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # ==================== Training Phase ====================
        model.train()
        train_loss = 0.0
        all_train_labels = []
        all_train_probs = []
        all_train_preds = []
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            # Get data
            images = batch['image'].to(device, dtype=torch.float32)
            labels = batch['label'].to(device, dtype=torch.float32).unsqueeze(1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = binary_focal_loss(outputs, labels, alpha=focal_alpha, gamma=focal_gamma)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            # Avg loss per item * batch_size
            train_loss += loss.item() * images.size(0)
            
            # Store predictions and labels for AUC and F1
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.detach().cpu().numpy()
            
            all_train_probs.extend(probs.flatten())
            all_train_preds.extend(preds.flatten())
            all_train_labels.extend(labels_np.flatten())
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Calculate epoch training metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_fbeta, epoch_train_recall, epoch_train_pr_auc = calculate_metrics(
            all_train_labels, all_train_probs, all_train_preds, fbeta_beta=fbeta_beta
        )
        
        # ==================== Validation Phase ====================
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_pbar:
                # Get data
                images = batch['image'].to(device)
                labels = batch['label'].to(device).float().unsqueeze(1)
                
                # Forward pass
                outputs = model(images)
                loss = binary_focal_loss(outputs, labels, alpha=focal_alpha, gamma=focal_gamma)
                
                # Calculate metrics
                val_loss += loss.item() * images.size(0)
                
                # Store predictions and labels for AUC and F1
                probs = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_val_probs.extend(probs.flatten())
                all_val_labels.extend(labels_np.flatten())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
        
        # Find the best threshold for this epoch based on validation data
        best_threshold, best_fbeta_at_threshold = find_best_threshold(
            all_val_labels, all_val_probs, beta=fbeta_beta
        )
        
        # print best thrshold for epoch 
        # Use the best threshold to create predictions
        all_val_preds = (np.array(all_val_probs) > best_threshold).astype(int)
        
        # Calculate epoch validation metrics (including the actual F-beta at the best threshold)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_fbeta, epoch_val_recall, epoch_val_pr_auc = calculate_metrics(
            all_val_labels, all_val_probs, all_val_preds, fbeta_beta=fbeta_beta
        )
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_fbeta'].append(epoch_train_fbeta)
        history['val_fbeta'].append(epoch_val_fbeta)
        history['train_recall'].append(epoch_train_recall)
        history['val_recall'].append(epoch_val_recall)
        history['train_pr_auc'].append(epoch_train_pr_auc)
        history['val_pr_auc'].append(epoch_val_pr_auc)
        history['best_thresholds'].append(best_threshold)
        
        # Print epoch results
        print(f"Train Loss: {epoch_train_loss:.4f} | Train F{fbeta_beta}: {epoch_train_fbeta:.4f} | Train Recall: {epoch_train_recall:.4f} | Train PR AUC: {epoch_train_pr_auc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val F{fbeta_beta}:   {epoch_val_fbeta:.4f} | Val Recall:   {epoch_val_recall:.4f} | Val PR AUC:   {epoch_val_pr_auc:.4f}")
        print(f"Best Threshold: {best_threshold:.4f} (F{fbeta_beta} at threshold: {best_fbeta_at_threshold:.4f})")
        
        # Step the scheduler based on validation PR AUC
        scheduler.step(epoch_val_pr_auc)
        
        # Save best model based on F-beta score (higher is better)
        if epoch_val_fbeta > best_val_fbeta:
            best_val_fbeta = epoch_val_fbeta
            best_val_loss = epoch_val_loss  # Also track loss for the best model
            checkpoint_path = save_path / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_fbeta': epoch_train_fbeta,
                'val_fbeta': epoch_val_fbeta,
                'train_recall': epoch_train_recall,
                'val_recall': epoch_val_recall,
                'train_pr_auc': epoch_train_pr_auc,
                'val_pr_auc': epoch_val_pr_auc,
                'best_threshold': best_threshold,
            }, checkpoint_path)
            print(f"Saved best model (val_f{fbeta_beta}: {best_val_fbeta:.4f}, val_loss: {epoch_val_loss:.4f}, val_recall: {epoch_val_recall:.4f}, val_pr_auc: {epoch_val_pr_auc:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_fbeta': epoch_train_fbeta,
                'val_fbeta': epoch_val_fbeta,
                'train_recall': epoch_train_recall,
                'val_recall': epoch_val_recall,
                'train_pr_auc': epoch_train_pr_auc,
                'val_pr_auc': epoch_val_pr_auc,
                'best_threshold': best_threshold,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    print("\nTraining completed!")
    print(f"Best validation F{fbeta_beta}: {best_val_fbeta:.4f} | Best validation loss: {best_val_loss:.4f}")
    
    
    return history



