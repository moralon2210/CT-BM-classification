# Brain Metastasis Detection from CT Scans

A deep learning pipeline for binary classification of brain metastases using CT imaging with EfficientNetV2-M architecture and multi-window preprocessing.

## Project Synopsis

This project implements an end-to-end training pipeline for detecting brain metastases from 2D CT scans. The model uses a pretrained **EfficientNetV2-M** backbone fine-tuned on CT images preprocessed with three specialized CT windows to maximize diagnostic signal extraction.

**Key Features:**
- **Automatic Data Validation**: Removes duplicates from csv (if exists) and validates image file existence
- **Triple CT Windowing**: Preprocesses CT scans using 3 clinical windows (Contrast Booster, Baseline, Edema Amplifier) to enhance different tissue characteristics
- **Focal Loss**: Handles class imbalance with configurable alpha/gamma parameters
- **F-beta Optimization**: Uses F2 score to prioritize recall and select threshold accordingly


## Environment Setup

### Prerequisites
- **Python**: 3.9

### Step 1: Create Conda Environment
```bash
conda create -n ct_bm python=3.9
conda activate ct_bm
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.0.0` + `torchvision>=0.15.0` (PyTorch + vision models)
- `monai>=1.3.0` (Medical image preprocessing)
- `scikit-learn>=1.3.0` (Metrics + data splitting)
- `pandas`, `numpy`, `matplotlib`, `tqdm`

### Step 4: Prepare Dataset Structure
Organize your data as follows:
```
CT-BM-classification/
├── Dataset/
│   ├── CTs/               # DICOM files (.dcm)
│   │   ├── patient001.dcm
│   │   ├── patient002.dcm
│   │   └── ...
│   └── labels1.csv        # Format: ID,Label (0=no metastasis, 1=metastasis)
├── checkpoints/           # Created automatically during training
├── src/
├── main.py
└── requirements.txt
```

**CSV Format:**
```csv
ID,Label
patient001,0
patient002,1
patient003,0
```

---

## Training Pipeline Execution

### Quick Start (Default Configuration)
```bash
python main.py
```

### Training Configuration
Edit `main.py` to customize hyperparameters:

```python
# Data settings
csv_path = "./Dataset/labels1.csv"
images_folder = "./Dataset/CTs"
df = df.iloc[:400]  # Remove this line to use full dataset

# DataLoader settings
batch_size = 4          # Adjust based on GPU memory
num_workers = 6         # Optimize for your CPU cores
prefetch_factor = 3     # Batches to prefetch per worker

# Training settings
num_epochs = 20         # Full training runs
learning_rate = 1e-4    # Adam optimizer LR
focal_gamma = 2.0       # Focal loss focusing parameter
```

### Training Process Overview

**Pipeline Steps (Automatic):**
1. **Data Validation**: 
   - Removes duplicate IDs
   - Validates image file existence
   - Saves cleaned CSV to `Dataset/labels_clean.csv`

2. **Data Splitting**:
   - Train: 70% | Validation: 15% | Test: 15%
   - Stratified by label to maintain class balance

3. **Focal Loss Alpha Calculation**:
   - α = (negative samples) / (total samples) from training set

4. **Dataset Creation**:
   - Applies triple-windowing + augmentation (train only)
   - Caches preprocessed images in RAM

5. **Training Loop**:
   - Optimizes F-beta (β=2) score
   - Finds optimal threshold per epoch using validation PR curve
   - Saves checkpoints:
     - `best_model.pth`: Best F-beta score model
     - `checkpoint_epoch_N.pth`: Every 5 epochs

### Monitoring Training

**Console Output:**
```
Epoch 1/20
----------------------------------------------------------
Train Loss: 0.2345 | Train F2: 0.7123 | Train Recall: 0.8234 | Train AP: 0.7891
Val Loss:   0.1987 | Val F2:   0.7456 | Val Recall:   0.8512 | Val AP:   0.8123
Best Threshold: 0.4523 (F2 at threshold: 0.7456)
✓ Saved best model (val_f2: 0.7456, val_loss: 0.1987, ...)
```

**Checkpoint Structure:**
```python
checkpoint = {
    'epoch': 15,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'val_fbeta': 0.8234,
    'val_recall': 0.8912,
    'val_avg_precision': 0.8567,
    'best_threshold': 0.4523
}
```

### Loading Trained Models

```python
from src.model import EfficientV2M
from src.train_loop import load_checkpoint
import torch

# Initialize model
model = EfficientV2M()

# Load best checkpoint
checkpoint_path = "./checkpoints/best_model.pth"
epoch = load_checkpoint(model, checkpoint_path, device='cuda')

# Load checkpoint with optimizer (for resuming training)
optimizer = torch.optim.Adam(model.parameters())
epoch = load_checkpoint(model, checkpoint_path, optimizer=optimizer)
```

---

## Project Structure

```
Samueli Institute/
├── src/
│   ├── data/
│   │   ├── data_processing.py    # Data validation & splitting
│   │   └── dataset.py            # MONAI transforms & dataloaders
│   ├── model.py                  # EfficientNetV2-M wrapper
│   ├── train_loop.py             # Training loop + focal loss + metrics
│   └── inference.py              # (Empty - for future inference scripts)
├── Dataset/
│   ├── CTs/                      # CT DICOM files
│   ├── labels1.csv               # Original labels
│   └── labels_clean.csv          # Cleaned labels (auto-generated)
├── checkpoints/                  # Model checkpoints (created during training)
├── main.py                       # Main training script
├── requirements.txt              # Python dependencies
├── activate_ct_bm.bat           # Windows conda activation script
└── README.md                     # This file
```

---

## Key Metrics Explanation

### F-beta Score (β=2)
```
F_beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
- **β=2**: Recall weighted 2× more than precision
- **Medical Rationale**: Missing metastases (false negatives) is more dangerous than false alarms

### Average Precision (AP)
- Area under the precision-recall curve
- Summarizes model performance across all thresholds
- Robust to class imbalance

### Threshold Optimization
- Default threshold (0.5) is suboptimal for imbalanced data
- Algorithm finds threshold maximizing F-beta on validation set per epoch
- Saved in checkpoint for inference consistency

---

## Common Issues & Solutions

### Issue: Low GPU Utilization
**Solution**: Increase `num_workers` (6-8) and `prefetch_factor` (3) in `main.py`:
```python
train_loader, val_loader, test_loader = create_dataloaders(
    ..., num_workers=8, prefetch_factor=3
)
```

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in `main.py`:
```python
train_loader, ... = create_dataloaders(..., batch_size=2)
```

### Issue: Slow First Epoch
**Explanation**: CacheDataset is preprocessing and caching images to RAM
**Solution**: Normal behavior - subsequent epochs will be 10-20× faster

### Issue: Import Errors on Windows
**Solution**: Use the provided activation script:
```bash
activate_ct_bm.bat
```

---

## Next Steps

### For Training:
1. Remove the `df = df.iloc[:400]` line in `main.py` to use full dataset
2. Increase `num_epochs` to 50-100 for production models
3. Experiment with learning rate schedules (e.g., ReduceLROnPlateau)

### For Inference:
1. Implement inference script in `src/inference.py`
2. Load best checkpoint + optimal threshold
3. Apply same triple-windowing transforms (no augmentation)

### For Evaluation:
1. Evaluate on held-out test set using best model
2. Generate confusion matrix and ROC curve
3. Visualize attention maps (Grad-CAM) for interpretability

---

## Citation & References

**Model Architecture:**
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021.

**Loss Function:**
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.

**Medical Imaging Framework:**
- MONAI Consortium (2020). MONAI: Medical Open Network for AI. https://monai.io/

---

## License & Contact

**Project**: Samueli Institute Brain Metastasis Detection  
**Author**: [Your Name]  
**Date**: February 2026

For questions or contributions, please contact [your-email@example.com]
