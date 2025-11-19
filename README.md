# Multimodal Deep Belief Network (iMDBN)

A PyTorch implementation of multimodal Deep Belief Networks for learning joint representations of images and categorical labels using Restricted Boltzmann Machines (RBMs).

## Features

- **Multimodal Learning**: Jointly model images and labels through a hierarchical RBM architecture
- **Cross-Modal Reconstruction**: Reconstruct images from labels and vice versa
- **Flexible Architecture**: Customizable layer sizes and hyperparameters
- **Robust Inference**: Noisy mean-field annealing and best-of-K search for high-quality reconstructions
- **WandB Integration**: Comprehensive logging with visualizations, PCA trajectories, and metrics
- **Compatible Saving**: Model format compatible with `numerical_analysis_pipeline`

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/multimodal-dbn.git
cd multimodal-dbn

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- scikit-learn
- tqdm
- PyYAML
- wandb (optional, for logging)

## Quick Start

### Training a Multimodal DBN

```python
from imdbn.models import iMDBN
from imdbn.datasets import create_dataloaders_uniform
import torch

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders_uniform(
    path2data="path/to/data",
    data_name="your_dataset",
    batch_size=64
)

# Define hyperparameters
params = {
    "LEARNING_RATE": 0.1,
    "WEIGHT_PENALTY": 1e-4,
    "INIT_MOMENTUM": 0.5,
    "FINAL_MOMENTUM": 0.95,
    "CD": 1,
    "EPOCHS_IMG": 100,
    "EPOCHS_JOINT": 150,
    "JOINT_LEARNING_RATE": 0.04,
    "CROSS_GIBBS_STEPS": 50,
    "JOINT_AUX_COND_STEPS": 30,
}

# Create model
model = iMDBN(
    layer_sizes_img=[10000, 1500, 500],  # Image DBN architecture
    layer_sizes_txt_or_joint=256,         # Joint layer size
    params=params,
    dataloader=train_loader,
    val_loader=val_loader,
    num_labels=32,                        # Number of classes
    device=torch.device("cuda")
)

# Train image DBN (layer-wise pretraining)
model.image_idbn.train(epochs=params["EPOCHS_IMG"])

# Initialize joint RBM biases
model.init_joint_bias_from_data()

# Train joint RBM (multimodal learning)
model.train_joint(
    epochs=params["EPOCHS_JOINT"],
    warmup_epochs=10,
    log_every_pca=10,
    log_every_metrics=5
)

# Save model
model.save_model("imdbn_trained.pkl")
```

### Cross-Modal Reconstruction

```python
# Load trained model
model_data = iMDBN.load_model("imdbn_trained.pkl")
model = iMDBN(...)  # Reinitialize with same architecture
# ... restore weights from model_data ...

# Reconstruct image from label
with torch.no_grad():
    images, labels = next(iter(test_loader))
    z_img = model.image_idbn.represent(images)

    # Image → Label prediction
    pred_labels, _ = model._cross_reconstruct(z_img, labels)

    # Label → Image generation
    _, recon_z = model._cross_reconstruct(z_img, labels)
    recon_images = model.image_idbn.decode(recon_z)
```

## Architecture

The iMDBN consists of three main components:

1. **Image DBN (iDBN)**: A stack of RBMs for learning hierarchical image features
   - Layer-wise greedy pretraining with Contrastive Divergence (CD)
   - Optional sparsity regularization on top layer
   - Supports PCA visualization and linear probes

2. **Joint RBM**: Connects image latent space with one-hot labels
   - Softmax group for label modeling (categorical distribution)
   - Auxiliary clamped-CD training for conditional distributions
   - μ-pull mechanism for class-conditional generation

3. **Cross-Modal Inference**:
   - **IMG→TXT**: Conditional Gibbs sampling
   - **TXT→IMG**: Noisy mean-field annealing + best-of-K search
   - Temperature scheduling and noise injection for robust reconstruction

## Configuration

Training hyperparameters can be specified via YAML config file:

```yaml
training:
  # Image DBN
  learning_rate: 0.1
  weight_penalty: 0.0001
  init_momentum: 0.5
  final_momentum: 0.95
  cd: 1
  epochs_image: 100

  # Joint RBM
  joint_learning_rate: 0.04
  epochs_joint: 150
  joint_cd: 1
  cross_gibbs_steps: 50

  # Auxiliary clamped-CD
  use_aux: true
  JOINT_AUX_COND_STEPS: 30
  JOINT_AUX_EVERY_K: 1
```

See `configs/multimodal_training_config.yaml` for a complete example.

## Training Script

Use the provided training script for end-to-end training:

```bash
python scripts/train_multimodal.py --config configs/multimodal_training_config.yaml
```

The script handles:
- Data loading
- Model initialization
- Layer-wise pretraining
- Joint RBM training
- Model saving
- WandB logging (optional)

## Model Saving Format

Models are saved in a **dual format** compatible with both analysis pipelines:

1. **DBN-compatible**: Has `"layers"` attribute for auto-detection by DBNAdapter
2. **Extended iMDBN**: Includes all components (`image_idbn`, `joint_rbm`, `features`, `metadata`)

```python
{
    "layers": [rbm1, rbm2, ..., joint_rbm],  # Flattened for DBNAdapter
    "params": {...},
    "image_idbn": iDBN_object,
    "joint_rbm": RBM_object,
    "features": {...},
    "z_class_mean": torch.Tensor,  # Per-class statistics
    "metadata": {...}
}
```

This format works seamlessly with:
- `numerical_analysis_pipeline` (DBNAdapter auto-detection)
- `groundeep-analysis` (Embedding_analysis)
- Custom analysis scripts

## Project Structure

```
multimodal-dbn/
├── imdbn/                   # Main package
│   ├── models/              # Model implementations
│   │   ├── rbm.py          # RBM with softmax groups
│   │   ├── idbn.py         # Image DBN
│   │   └── imdbn.py        # Multimodal DBN
│   ├── utils/               # Utilities
│   │   ├── logging.py      # Visualization and logging
│   │   └── conditional_steps.py  # Conditional training utils
│   └── datasets/            # Data loaders
│       ├── uniform_dataset.py
│       └── zipfian_dataset.py
├── configs/                 # Configuration files
├── scripts/                 # Training scripts
├── examples/                # Example usage
└── tests/                   # Unit tests
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_dbn,
  author = {Your Name},
  title = {Multimodal Deep Belief Networks},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/multimodal-dbn}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This implementation builds on research in:
- Deep Belief Networks (Hinton et al., 2006)
- Multimodal learning with RBMs (Ngiam et al., 2011)
- Contrastive Divergence training (Hinton, 2002)
