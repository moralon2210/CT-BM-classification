from sklearn.metrics import fbeta_score, recall_score, auc, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

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
    Find the optimal threshold that maximizes the F-beta score.
    Tests multiple thresholds and calculates actual F-beta score for each.
    
    """
    labels = np.array(labels)
    probs = np.array(probs)
    
    # Test thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.0, 0.02)
    best_fbeta = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        # Calculate F-beta score using sklearn (handles edge cases properly)
        fbeta = fbeta_score(labels, preds, beta=beta, zero_division=0)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
    
    return best_threshold, best_fbeta


def calculate_metrics(labels, probs, preds, fbeta_beta=2.0):
    """
    Calculate classification metrics: F-beta score, recall, and PR AUC.
    
    """
    fbeta = fbeta_score(labels, preds, beta=fbeta_beta)
    recall = recall_score(labels, preds)
    
    # Calculate PR AUC (Precision-Recall Area Under Curve)
    precision, recall_curve, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall_curve, precision)
    
    return fbeta, recall, pr_auc