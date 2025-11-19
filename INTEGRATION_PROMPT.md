# Integration Prompt for numerical_analysis_pipeline

## Objective

Add support for **multimodal Deep Belief Networks (iMDBN)** to the `numerical_analysis_pipeline` repository by creating an iMDBNAdapter that extends the existing adapter system.

---

## Background

The `numerical_analysis_pipeline` currently supports:
- **DBNs** via `DBNAdapter` (auto-detected when model has `.layers` attribute)
- **VAEs** via `VAEAdapter` (auto-detected when model has `.encoder` and `.decoder`)
- **Generic PyTorch** via `PyTorchAdapter` (fallback)

The **iMDBN (image Multimodal Deep Belief Network)** is a hierarchical model with:
1. **Image DBN (iDBN)**: Stack of RBMs for hierarchical image features
2. **Joint RBM**: Connects image latent space with one-hot labels via softmax groups
3. **Cross-modal inference**: Bidirectional reconstruction between images and labels

---

## Task

Create an **iMDBNAdapter** that provides:

1. **Auto-detection** in `create_adapter()` function
2. **Encoding** methods for extracting multimodal embeddings
3. **Decoding** for cross-modal reconstruction
4. **Layer-wise encoding** for analyzing intermediate representations

---

## Implementation Details

### 1. File Location

Create the adapter at:
```
groundeep_analysis/core/adapters/imdbn_adapter.py
```

### 2. Adapter Class Structure

```python
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter


class iMDBNAdapter(BaseAdapter):
    """
    Adapter for image Multimodal Deep Belief Networks (iMDBN).

    Supports:
    - Joint multimodal embeddings (image + label)
    - Image-only embeddings (before joint layer)
    - Layer-wise embeddings (each RBM layer)
    - Cross-modal reconstruction (image ↔ label)
    """

    def __init__(self, model, device: str = "auto", **kwargs):
        super().__init__(model, device, **kwargs)

        # Extract components from saved model dict
        if isinstance(model, dict):
            self.image_idbn = model.get("image_idbn")
            self.joint_rbm = model.get("joint_rbm")
            self.num_labels = model.get("num_labels", 32)
            self.Dz_img = model.get("Dz_img")

            # Optional components
            self.z_class_mean = model.get("z_class_mean", None)
            self.features = model.get("features", None)
            self.metadata = model.get("metadata", {})
        else:
            # Model is an iMDBN object
            self.image_idbn = model.image_idbn
            self.joint_rbm = model.joint_rbm
            self.num_labels = model.num_labels
            self.Dz_img = model.Dz_img
            self.z_class_mean = getattr(model, "z_class_mean", None)
            self.features = getattr(model, "features", None)
            self.metadata = getattr(model, "metadata", {})

        # Move to device
        self._move_to_device()

    def _move_to_device(self):
        """Move model components to target device."""
        device = self.device

        # Move image DBN layers
        if hasattr(self.image_idbn, 'layers'):
            for rbm in self.image_idbn.layers:
                rbm.W = rbm.W.to(device)
                rbm.hid_bias = rbm.hid_bias.to(device)
                rbm.vis_bias = rbm.vis_bias.to(device)

        # Move joint RBM
        self.joint_rbm.W = self.joint_rbm.W.to(device)
        self.joint_rbm.hid_bias = self.joint_rbm.hid_bias.to(device)
        self.joint_rbm.vis_bias = self.joint_rbm.vis_bias.to(device)

        # Move statistics
        if self.z_class_mean is not None:
            self.z_class_mean = self.z_class_mean.to(device)

    def encode(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Extract joint multimodal embeddings (top layer of joint RBM).

        Args:
            batch: Tuple of (images, labels)
                - images: [B, C, H, W] or [B, D]
                - labels: [B, num_labels] one-hot encoded

        Returns:
            Joint embeddings [B, joint_hidden_dim]
        """
        images, labels = batch

        # Prepare inputs
        images = self.prepare_input(images).to(self.device).view(images.size(0), -1).float()
        labels = self.prepare_input(labels).to(self.device).float()

        with torch.no_grad():
            # Extract image embeddings
            z_img = self._encode_image_only(images)

            # Concatenate with labels
            v_joint = torch.cat([z_img, labels], dim=1)

            # Pass through joint RBM
            h_joint = self.joint_rbm.forward(v_joint)

        return h_joint

    def _encode_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image-only embeddings (before joint layer)."""
        with torch.no_grad():
            v = images.view(images.size(0), -1).float()
            for rbm in self.image_idbn.layers:
                v = rbm.forward(v)
        return v

    def encode_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """
        Public method: Extract image-only embeddings.

        Args:
            images: [B, C, H, W] or [B, D]

        Returns:
            Image embeddings [B, Dz_img]
        """
        images = self.prepare_input(images).to(self.device)
        return self._encode_image_only(images)

    def encode_layerwise(self, batch, layers: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Extract embeddings at each layer of the hierarchy.

        Args:
            batch: Tuple of (images, labels) or just images
            layers: Specific layers to extract (1-indexed). If None, extract all.

        Returns:
            List of embeddings at each layer
        """
        # Handle both (images, labels) and images-only
        if isinstance(batch, tuple):
            images, labels = batch
        else:
            images = batch
            labels = None

        images = self.prepare_input(images).to(self.device).view(images.size(0), -1).float()

        embeddings = []
        v = images

        with torch.no_grad():
            # Image DBN layers
            for i, rbm in enumerate(self.image_idbn.layers, start=1):
                v = rbm.forward(v)
                if layers is None or i in layers:
                    embeddings.append(v.clone())

            # Joint layer (if labels provided)
            if labels is not None:
                labels = self.prepare_input(labels).to(self.device).float()
                v_joint = torch.cat([v, labels], dim=1)
                h_joint = self.joint_rbm.forward(v_joint)

                joint_layer_idx = len(self.image_idbn.layers) + 1
                if layers is None or joint_layer_idx in layers:
                    embeddings.append(h_joint)

        return embeddings

    def decode(self, z: torch.Tensor, decode_images: bool = True) -> torch.Tensor:
        """
        Decode embeddings back to input space.

        Args:
            z: Latent embeddings (either joint or image-only)
            decode_images: If True, decode all the way to images

        Returns:
            Reconstructed images [B, D]
        """
        z = self.prepare_input(z).to(self.device)

        with torch.no_grad():
            # If z is from joint layer, first decode to image latent space
            if z.size(1) == self.joint_rbm.num_hidden:
                # Decode joint → joint visible
                v_joint = self.joint_rbm.backward(z)
                # Extract image part
                z_img = v_joint[:, :self.Dz_img]
            else:
                z_img = z

            # Decode through image DBN if requested
            if decode_images:
                cur = z_img
                for rbm in reversed(self.image_idbn.layers):
                    cur = rbm.backward(cur)
                return cur
            else:
                return z_img

    def get_num_layers(self) -> int:
        """Return total number of layers (image DBN + joint)."""
        return len(self.image_idbn.layers) + 1

    def get_layer_sizes(self) -> List[int]:
        """Return sizes of all layers."""
        sizes = []
        # Image DBN layers
        for rbm in self.image_idbn.layers:
            sizes.append(rbm.num_hidden)
        # Joint layer
        sizes.append(self.joint_rbm.num_hidden)
        return sizes
```

### 3. Auto-Detection in `__init__.py`

Update `groundeep_analysis/core/adapters/__init__.py`:

```python
from .imdbn_adapter import iMDBNAdapter

# In create_adapter() function, add before DBN detection:

def create_adapter(model, adapter_type="auto", device="auto", **kwargs):
    """
    Create appropriate adapter for the model.

    Supports: dbn, vae, imdbn, pytorch, or "auto" for automatic detection.
    """

    if adapter_type == "imdbn":
        return iMDBNAdapter(model, device=device, **kwargs)

    if adapter_type == "auto":
        # 1. Check for iMDBN (has both image_idbn and joint_rbm)
        if isinstance(model, dict):
            if "image_idbn" in model and "joint_rbm" in model:
                logger.info("Auto-detected iMDBN model")
                return iMDBNAdapter(model, device=device, **kwargs)
        elif hasattr(model, "image_idbn") and hasattr(model, "joint_rbm"):
            logger.info("Auto-detected iMDBN model")
            return iMDBNAdapter(model, device=device, **kwargs)

        # 2. Check for DBN (existing logic)
        # ... rest of auto-detection ...
```

### 4. Update Exports

In `groundeep_analysis/core/adapters/__init__.py`:

```python
__all__ = [
    "BaseAdapter",
    "DBNAdapter",
    "VAEAdapter",
    "PyTorchAdapter",
    "iMDBNAdapter",  # Add this
    "create_adapter",
]
```

---

## Testing

Create `tests/test_imdbn_adapter.py`:

```python
import torch
import pytest
from groundeep_analysis.core.adapters import create_adapter, iMDBNAdapter


def test_imdbn_auto_detection():
    """Test that iMDBN models are auto-detected."""
    # Mock iMDBN model structure
    mock_model = {
        "image_idbn": MockiDBN(),
        "joint_rbm": MockRBM(532, 256),  # 500 + 32 → 256
        "num_labels": 32,
        "Dz_img": 500,
        "metadata": {"model_type": "iMDBN"}
    }

    adapter = create_adapter(mock_model, adapter_type="auto")
    assert isinstance(adapter, iMDBNAdapter)


def test_imdbn_encode():
    """Test encoding with iMDBNAdapter."""
    adapter = create_adapter(mock_imdbn_model)

    # Create dummy batch
    images = torch.randn(8, 1, 100, 100)
    labels = torch.zeros(8, 32)
    labels[torch.arange(8), torch.randint(0, 32, (8,))] = 1.0

    # Encode
    embeddings = adapter.encode((images, labels))

    assert embeddings.shape == (8, 256)  # Joint hidden dim


def test_imdbn_layerwise():
    """Test layer-wise encoding."""
    adapter = create_adapter(mock_imdbn_model)

    images = torch.randn(8, 1, 100, 100)
    labels = torch.zeros(8, 32)
    labels[torch.arange(8), torch.randint(0, 32, (8,))] = 1.0

    # Get all layers
    layer_embeddings = adapter.encode_layerwise((images, labels))

    # Should have: [layer1, layer2, ..., joint]
    assert len(layer_embeddings) == adapter.get_num_layers()
```

---

## Integration Checklist

- [ ] Create `imdbn_adapter.py` with complete iMDBNAdapter class
- [ ] Update `__init__.py` with auto-detection logic
- [ ] Add iMDBNAdapter to exports
- [ ] Create unit tests in `tests/test_imdbn_adapter.py`
- [ ] Run existing test suite to ensure no regressions
- [ ] Update documentation/README with iMDBN support
- [ ] (Optional) Add example notebook showing iMDBN analysis

---

## Expected Behavior

After integration, users should be able to:

```python
from groundeep_analysis.core.adapters import create_adapter
import pickle

# Load iMDBN model (saved with multimodal-dbn package)
with open("imdbn_trained.pkl", "rb") as f:
    model_data = pickle.load(f)

# Auto-detect and create adapter
adapter = create_adapter(model_data, adapter_type="auto")  # Detects iMDBN

# Extract embeddings
images, labels = get_test_batch()
joint_embeddings = adapter.encode((images, labels))  # Joint multimodal
image_embeddings = adapter.encode_image_only(images)  # Image-only

# Layer-wise analysis
layer_embeddings = adapter.encode_layerwise((images, labels))

# Reconstruction
reconstructed = adapter.decode(joint_embeddings)
```

---

## Notes

1. **Dual format compatibility**: iMDBN models are saved with both `"layers"` (for DBN compatibility) and full iMDBN structure. The iMDBNAdapter should detect the full structure first to enable multimodal-specific features.

2. **Image-only vs Joint embeddings**: The adapter provides both `encode()` (joint) and `encode_image_only()` (image-only) to support different analysis workflows.

3. **Cross-modal features**: Future extensions could add:
   - `cross_reconstruct()` for image↔label reconstruction
   - `predict_labels()` for classification
   - `generate_from_label()` for conditional generation

4. **Backward compatibility**: The existing DBNAdapter should still work with iMDBN models (treating them as flat DBNs), but iMDBNAdapter provides multimodal-specific functionality.

---

## Questions?

If you need clarification on:
- The iMDBN model architecture
- Specific methods/attributes available
- Integration with existing pipeline features

Please refer to the `multimodal-dbn` repository or contact the maintainer.
