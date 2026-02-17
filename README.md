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

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd CT-BM-classification
```

### Step 2: Create Conda Environment
```bash
conda create -n ct_bm python=3.9
conda activate ct_bm
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```


### Step 4: Prepare Dataset Structure
Organize your data as follows:
```
Samueli Institute/
├── Dataset/
│   ├── train/
│   │   ├── CTs/           # Training DICOM files
│   │   │   ├── patient001.dcm
│   │   │   ├── patient002.dcm
│   │   │   └── ...
│   │   └── labels1.csv    # Format: ID,Label (0=no metastasis, 1=metastasis)
│   └── inference/
│       └── CTs/           # Inference images (no labels needed)
├── checkpoints/           # Created automatically during training
├── results/               # Created automatically for logs and metrics
├── src/
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

## Running the Pipeline

After completing the environment setup and dataset preparation, you can run the pipeline in two modes using `main.py`:

### Mode 1: Training Mode
Train a new model from scratch > uses the data from Dataset/train

```bash
python main.py train
```
This will:
- Save the best model to `checkpoints/best_model.pth`
- Save training history and metrics plots to `results/`
- Evaluate on test set and save:
  - `results/test_results.csv` - F2-score, recall, and precision metrics
  - `results/test_predictions.csv` - Detailed predictions with image names, true labels, probabilities, and predictions
  - `results/confusion_matrix.csv` - Confusion matrix data
  - `results/confusion_matrix.png` - Confusion matrix visualization

### Mode 2: Inference Mode
Run predictions on new CT images using a trained model > uses the data from Dataset/inference

```bash
python main.py inference
```

This will:
- Load the best model from `checkpoints/best_model.pth`
- Save predictions to `results/predictions.csv`


## Training Configuration

# Key parameters
batch_size = 8          # Adjust based on GPU memory
num_epochs = 20         # Increase for production
learning_rate = 1e-4    # reduced during training using a scheduler
```

## Project Structure

```
Samueli Institute/
├── src/
│   ├── data/              # Data processing and loading
│   ├── train/             # Training scripts and utilities
│   ├── model.py           # EfficientNetV2-M wrapper
│   └── inference.py       # Inference pipeline
├── Dataset/
│   ├── train/             # Training files
│   └── inference/CTs/     # Inference CT files
├── checkpoints/           # Saved models
├── results/               # Training logs and metrics
└── main.py                # Entry point
```



