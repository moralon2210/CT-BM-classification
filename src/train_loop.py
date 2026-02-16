import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import fbeta_score, recall_score, average_precision_score, precision_recall_curve
import torch.nn.functional as F

def binary_focal_loss(logits, targets, alpha=0.75, gamma=2.0, reduction='mean'):
    """
    Binary Focal Loss implementation.

    """
    # Ensure targets are floats for calculation
    targets = targets.float()
    
    # Calculate Binary Cross Entropy Loss
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    
    # Calculate p_t (probability of the true class)
    # If target=1, p_t = p. If target=0, p_t = 1 - p.
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Calculate alpha_t (class weighting factor)
    # If target=1, alpha_t = alpha. If target=0, alpha_t = 1 - alpha.
    # This balances the contribution of positive vs negative examples
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Apply Focal Loss formula: alpha_t * (1 - p_t)^gamma * CE_loss
    # The (1 - p_t)^gamma term down-weights easy examples
    loss = alpha_t * (1 - p_t)**gamma * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

    return loss

def find_best_threshold(labels, probs, beta=2.0):
    """
    Find the optimal threshold that maximizes the F-beta score using precision-recall curve.
    
    """
    labels = np.array(labels)
    probs = np.array(probs)
    
    # Get precision, recall, and thresholds for all unique probability values
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    
    # Calculate F-beta score for each threshold
    # F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    beta_squared = beta ** 2
    
    # Avoid division by zero
    denominator = beta_squared * precision[:-1] + recall[:-1]
    fbeta_scores = np.where(
        denominator == 0,
        0,
        (1 + beta_squared) * (precision[:-1] * recall[:-1]) / denominator
    )
    
    # Find the threshold that maximizes F-beta score
    best_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_idx]
    best_fbeta = fbeta_scores[best_idx]
    
    return best_threshold, best_fbeta


def calculate_metrics(labels, probs, preds, fbeta_beta=2.0):
    """
    Calculate classification metrics: F-beta score, recall, and Average Precision.
    
    """
    fbeta = fbeta_score(labels, preds, beta=fbeta_beta)
    recall = recall_score(labels, preds)
    avg_precision = average_precision_score(labels, probs)
    
    return fbeta, recall, avg_precision

def train_loop(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, 
               device=None, save_dir="./checkpoints", focal_alpha=0.75, focal_gamma=2.0, 
               fbeta_beta=2.0):
    """
    Training loop for the model with Focal Loss for handling class imbalance

    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup loss function (Custom Binary Focal Loss)
    # Alpha: weighting for positive class (higher = more weight on positives)
    # Gamma: focusing parameter (higher = focus more on hard examples)
    print(f"Using custom binary focal loss with alpha={focal_alpha}, gamma={focal_gamma}")
    
    # Create checkpoint directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_fbeta': [],
        'val_fbeta': [],
        'train_recall': [],
        'val_recall': [],
        'train_avg_precision': [],
        'val_avg_precision': [],
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
        epoch_train_fbeta, epoch_train_recall, epoch_train_avg_precision = calculate_metrics(
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
        
        # Use the best threshold to create predictions
        all_val_preds = (np.array(all_val_probs) > best_threshold).astype(int)
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_fbeta, epoch_val_recall, epoch_val_avg_precision = calculate_metrics(
            all_val_labels, all_val_probs, all_val_preds, fbeta_beta=fbeta_beta
        )
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_fbeta'].append(epoch_train_fbeta)
        history['val_fbeta'].append(epoch_val_fbeta)
        history['train_recall'].append(epoch_train_recall)
        history['val_recall'].append(epoch_val_recall)
        history['train_avg_precision'].append(epoch_train_avg_precision)
        history['val_avg_precision'].append(epoch_val_avg_precision)
        history['best_thresholds'].append(best_threshold)
        
        # Print epoch results
        print(f"Train Loss: {epoch_train_loss:.4f} | Train F{fbeta_beta}: {epoch_train_fbeta:.4f} | Train Recall: {epoch_train_recall:.4f} | Train AP: {epoch_train_avg_precision:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val F{fbeta_beta}:   {epoch_val_fbeta:.4f} | Val Recall:   {epoch_val_recall:.4f} | Val AP:   {epoch_val_avg_precision:.4f}")
        print(f"Best Threshold: {best_threshold:.4f} (F{fbeta_beta} at threshold: {best_fbeta_at_threshold:.4f})")
        
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
                'train_avg_precision': epoch_train_avg_precision,
                'val_avg_precision': epoch_val_avg_precision,
                'best_threshold': best_threshold,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_f{fbeta_beta}: {best_val_fbeta:.4f}, val_loss: {epoch_val_loss:.4f}, val_recall: {epoch_val_recall:.4f}, val_ap: {epoch_val_avg_precision:.4f})")
        
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
                'train_avg_precision': epoch_train_avg_precision,
                'val_avg_precision': epoch_val_avg_precision,
                'best_threshold': best_threshold,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation F{fbeta_beta} score: {best_val_fbeta:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    return history


def load_checkpoint(model, checkpoint_path, optimizer=None, device=None):
    """
    Load model from checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load model on
    
    Returns:
        epoch: Epoch number from checkpoint
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch
