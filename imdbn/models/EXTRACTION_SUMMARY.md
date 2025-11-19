# iDBN and iMDBN Extraction Summary

This document describes the clean, standalone extraction of the `iDBN` and `iMDBN` classes from the original Groundeep project.

## Extraction Details

### Source Files
- **Source**: `/home/student/Desktop/Groundeep/src/classes/gdbn_model.py`
- **Extraction Date**: 2025-11-19

### Extracted Classes

#### 1. iDBN (Image Deep Belief Network)
**File**: `idbn.py` (373 lines)

A specialized Deep Belief Network for image feature learning through greedy layer-wise training.

**Key Components**:
- `__init__`: Initialize layers, parameters, and validation features
- `train()`: Layer-wise greedy pretraining with RBMs
  - Auto-reconstruction logging every 5 epochs
  - PCA embedding visualization
  - Linear probe evaluation on learned representations
- `represent()`: Forward pass to extract features at any layer
- `reconstruct()`: Encode and decode for image reconstruction
- `decode()`: Decode from top-layer latents back to image space
- `save_model()`: Pickle serialization of layers and parameters

**Features**:
- Configurable layer sizes and RBM parameters
- Dynamic learning rate decay
- Sparsity regularization on top layer (optional)
- Per-layer monitoring of specific layers
- W&B logging integration
- Validation feature extraction (labels, geometry, density)

**Dependencies**:
- `imdbn.models.rbm.RBM`: Base RBM implementation
- `imdbn.utils.wandb_utils`: W&B visualization functions
- `imdbn.utils.probe_utils`: Linear probe evaluation
- `torch`, `numpy`, `sklearn.decomposition.PCA`, `tqdm`, `wandb`

#### 2. iMDBN (Image-Multimodal Deep Belief Network)
**File**: `imdbn.py` (934 lines)

A joint multimodal model connecting image features and discrete class labels for cross-modal learning.

**Key Components**:
- `__init__`: Initialize image iDBN and joint RBM
- `_build_joint()`: Create joint RBM with softmax-group for class labels
- `train_joint()`: Multimodal training with two phases:
  - **Warmup** (8 epochs): Label clamping only (supervised initialization)
  - **Main** (remaining epochs): Free CD + auxiliary objectives
- `_cross_reconstruct()`: Cross-modal reconstruction with μ-pull regularization
  - IMG→TXT: Image latents → label distribution
  - TXT→IMG: Labels → reconstructed images with per-class priors
- `init_joint_bias_from_data()`: Initialize biases from data statistics
- `load_pretrained_image_idbn()`: Load pre-trained image feature extractor
- `finetune_image_last_layer()`: Light fine-tuning of image network
- `represent()`: Extract joint latent codes
- `save_model()`: Save complete model (DBN-compatible + extended format)
- `load_model()`: Static method to load saved models
- `_log_snapshots()`: W&B logging of reconstruction quality and metrics

**Features**:
- Flexible constructor: supports both long-form and short-form signatures
- Per-class latent statistics (z_class_mean) for regularization
- Noisy mean-field annealing for cross-modal inference
- Best-of-K refinement for improved reconstructions
- Comprehensive W&B logging:
  - Cross-modal metrics (top-1/3 accuracy, CE, MSE)
  - PCA embeddings of joint latents
  - Linear probe performance
  - Reconstruction grids and confusion matrices
- Optional affine gain-control for latent scaling
- Support for class names in visualizations

**Training Strategy**:
- **Warmup Phase** (epochs 0-7): Label clamping (y-clamp) with 2 update passes per batch
- **Main Phase** (epoch 8+):
  - Free CD on full [z_img, y] data
  - Auxiliary y-clamp: 1 per batch (labeled reconstruction)
  - Auxiliary z-clamp: every 50 batches (image consistency)
  - Free-form label inference (IMG→TXT)
  - Per-class guided reconstruction (TXT→IMG with μ-pull)

**Model Saving Format**:
Saves in dual format for compatibility:
1. **DBN-compatible**: `layers` key with flattened RBM list (auto-detection)
2. **Extended iMDBN**: All components for full functionality
   - `image_idbn`: Complete iDBN object
   - `joint_rbm`: Joint RBM layer
   - `z_class_mean`: Per-class latent statistics
   - `features`: Validation metadata
   - `metadata`: Timestamp and info

**Dependencies**:
- `imdbn.models.idbn.iDBN`: Image feature extractor
- `imdbn.models.rbm.RBM`: RBM implementation
- `imdbn.utils.wandb_utils`: W&B visualization
- `imdbn.utils.probe_utils`: Linear probe and embedding extraction
- `torch`, `numpy`, `sklearn.decomposition.PCA`, `tqdm`, `wandb`

## Import Updates

All imports have been updated to use the new package structure:
```python
# Old (Groundeep)
from src.utils.wandb_utils import ...
from src.utils.probe_utils import ...

# New (iMDBN)
from imdbn.utils.wandb_utils import ...
from imdbn.utils.probe_utils import ...
```

## File Organization

```
/home/student/Desktop/multimodal-dbn/
└── imdbn/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── rbm.py           (existing - RBM base class)
    │   ├── idbn.py          (NEW - Image DBN)
    │   ├── imdbn.py         (NEW - Multimodal DBN)
    │   └── EXTRACTION_SUMMARY.md  (this file)
    ├── utils/
    │   ├── __init__.py
    │   ├── wandb_utils.py          (copied from Groundeep)
    │   ├── probe_utils.py          (copied from Groundeep)
    │   ├── imdbn_logging.py        (copied from Groundeep)
    │   ├── energy_utils.py         (copied from Groundeep)
    │   └── conditional_steps.py    (copied from Groundeep)
    ├── datasets/
    └── __init__.py
```

## Usage Example

### Training Image DBN
```python
import torch
from imdbn.models import iDBN

# Prepare data
dataloader = ...  # torch.utils.data.DataLoader
val_loader = ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
params = {
    "LEARNING_RATE": 0.1,
    "WEIGHT_PENALTY": 0.0001,
    "INIT_MOMENTUM": 0.5,
    "FINAL_MOMENTUM": 0.95,
    "LEARNING_RATE_DYNAMIC": True,
    "CD": 1,
    "SPARSITY": False,
    "SPARSITY_FACTOR": 0.1,
}

# Create and train
model = iDBN(
    layer_sizes=[784, 500, 200, 30],
    params=params,
    dataloader=dataloader,
    val_loader=val_loader,
    device=device,
    wandb_run=None,
)

model.train(epochs=100, log_every_pca=25, log_every_probe=10)
model.save_model("idbn_model.pkl")
```

### Training Multimodal DBN
```python
import torch
from imdbn.models import iMDBN

# Prepare data (with one-hot labels)
dataloader = ...
val_loader = ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "LEARNING_RATE": 0.1,
    "WEIGHT_PENALTY": 0.0001,
    "INIT_MOMENTUM": 0.5,
    "FINAL_MOMENTUM": 0.95,
    "LEARNING_RATE_DYNAMIC": True,
    "CD": 1,
    "JOINT_CD": 1,
    "JOINT_LEARNING_RATE": 0.1,
    "CROSS_GIBBS_STEPS": 50,
    "JOINT_AUX_COND_STEPS": 10,
}

# Create model
model = iMDBN(
    layer_sizes_img=[784, 500, 200, 30],
    joint_layer_size=256,
    params=params,
    dataloader=dataloader,
    val_loader=val_loader,
    device=device,
    num_labels=32,
)

# Load pre-trained image features (optional)
model.load_pretrained_image_idbn("idbn_model.pkl")

# Train jointly
model.train_joint(epochs=100)
model.save_model("imdbn_model.pkl")

# Load later
loaded_data = iMDBN.load_model("imdbn_model.pkl", device=device)
image_idbn = loaded_data["image_idbn"]
joint_rbm = loaded_data["joint_rbm"]
```

## Code Quality Improvements

1. **Clean Imports**: Removed unused imports (energy_utils, conditional_steps for iDBN)
2. **Documentation**: Added comprehensive docstrings to all classes and methods
3. **Type Hints**: Added Python type hints for better IDE support
4. **Modularity**: Separated concerns into focused classes
5. **Error Handling**: Robust fallbacks for missing features/attributes
6. **Device Management**: Proper device placement for CUDA/CPU flexibility

## Key Design Decisions

1. **Dual Save Format**: Saves both DBN-compatible and extended iMDBN formats for flexibility
2. **Per-Class Statistics**: Computes z_class_mean for informed cross-modal reconstruction
3. **Warmup Phase**: Strong supervision in early training ensures good initialization
4. **Auxiliary Objectives**: Multiple clamping strategies prevent mode collapse
5. **μ-Pull Regularization**: Guides latent reconstruction toward class-specific means

## Testing Notes

- All imports verified and working correctly
- Both classes can be instantiated independently
- Cross-module dependencies properly resolved
- Package-level imports functional

## Lines of Code

- `idbn.py`: 373 lines (clean, focused)
- `imdbn.py`: 934 lines (comprehensive with all methods)
- Total extracted: 1,307 lines

Original source `gdbn_model.py`: 1,297 lines (including RBM, helpers, and unused imports)
