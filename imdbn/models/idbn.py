"""
iDBN (Image Deep Belief Network)

A multilayer Deep Belief Network specialized for image data.
Supports layer-wise pretraining with RBMs, PCA visualization, linear probes, and reconstruction.

Key Features:
- Greedy layer-wise training of stacked RBMs
- Auto-reconstruction logging during training
- PCA-based embedding visualization
- Linear probe evaluation on learned representations
- Support for sparsity regularization on top layer
- Configurable logging via YAML
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from sklearn.decomposition import PCA

from imdbn.models.rbm import RBM
from imdbn.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
from imdbn.utils.probe_utils import (
    log_linear_probe,
    compute_val_embeddings_and_features,
)


class iDBN:
    """
    Image Deep Belief Network (iDBN).

    A stack of RBMs trained layer-by-layer on image data with support for:
    - Greedy layer-wise pretraining
    - Visualization of learned representations via PCA
    - Linear probe evaluation on intermediate layers
    - Auto-reconstruction quality monitoring

    Attributes:
        layers: List of RBM layers
        params: Training hyperparameters dictionary
        dataloader: Training data loader
        val_loader: Validation data loader
        device: Torch device (cuda/cpu)
        wandb_run: Optional W&B run object for logging
        features: Validation set features (labels, geometry measures, etc.)
        arch_str: Architecture string for logging (e.g., "784-500-200-30")
        arch_dir: Directory for architecture-specific outputs
    """

    def __init__(
        self,
        layer_sizes: List[int],
        params: dict,
        dataloader,
        val_loader,
        device,
        wandb_run=None,
        logging_config_path: Optional[str] = None
    ):
        """
        Initialize iDBN.

        Args:
            layer_sizes: List of layer dimensions, e.g., [784, 500, 200, 30]
            params: Configuration dict with keys:
                - LEARNING_RATE: Float, default 0.1
                - WEIGHT_PENALTY: Float, default 0.0001
                - INIT_MOMENTUM: Float, default 0.5
                - FINAL_MOMENTUM: Float, default 0.95
                - LEARNING_RATE_DYNAMIC: Bool, dynamic LR decay
                - CD: Int, contrastive divergence steps (default 1)
                - SPARSITY: Bool, enable sparsity on top layer
                - SPARSITY_FACTOR: Float, target activation level
            dataloader: Training DataLoader
            val_loader: Validation DataLoader
            device: torch.device
            wandb_run: Optional W&B run for logging
            logging_config_path: Optional path to logging config YAML
        """
        self.layers: List[RBM] = []
        self.params = params
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run

        # Load logging config if present
        self.logging_cfg = {}
        try:
            import yaml
            from pathlib import Path
            cfg_path = Path(logging_config_path) if logging_config_path else Path("src/configs/logging_config.yaml")
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    self.logging_cfg = cfg
        except Exception:
            pass

        # Fields expected by utility functions
        self.text_flag = False
        self.arch_str = "-".join(map(str, layer_sizes))
        self.arch_dir = os.path.join("logs-idbn", f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self.cd_k = int(self.params.get("CD", 1))
        self.sparsity_last = bool(self.params.get("SPARSITY", False))
        self.sparsity_factor = float(self.params.get("SPARSITY_FACTOR", 0.1))

        # Cache validation batch
        try:
            self.val_batch, self.val_labels = next(iter(val_loader))
        except Exception:
            self.val_batch, self.val_labels = None, None

        # Extract validation features (labels, geometry, density)
        self.features = None
        try:
            indices = val_loader.dataset.indices
            base = val_loader.dataset.dataset
            numeric_labels = torch.tensor([base.labels[i] for i in indices], dtype=torch.float32)
            cumArea_vals = [base.cumArea_list[i] for i in indices]
            convex_hull = [base.CH_list[i] for i in indices]
            density_src = getattr(base, "density_list", None)
            density_vals = [density_src[i] for i in indices] if density_src is not None else None
            self.features = {
                "Cumulative Area": torch.tensor(cumArea_vals, dtype=torch.float32),
                "Convex Hull": torch.tensor(convex_hull, dtype=torch.float32),
                "Labels": numeric_labels,
            }
            if density_vals is not None:
                self.features["Density"] = torch.tensor(density_vals, dtype=torch.float32)
        except Exception:
            pass

        # Build RBM layers
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=self.params["LEARNING_RATE"],
                weight_decay=self.params["WEIGHT_PENALTY"],
                momentum=self.params["INIT_MOMENTUM"],
                dynamic_lr=self.params["LEARNING_RATE_DYNAMIC"],
                final_momentum=self.params["FINAL_MOMENTUM"],
                sparsity=(self.sparsity_last and i == len(layer_sizes) - 2),
                sparsity_factor=self.sparsity_factor,
            ).to(self.device)
            self.layers.append(rbm)

    def _layers_to_monitor(self) -> List[int]:
        """
        Determine which layers to monitor during training.

        Returns:
            List of layer indices to log (top layer and first if multiple layers)
        """
        layers = {len(self.layers)}
        if len(self.layers) > 1:
            layers.add(1)
        return sorted(layers)

    def _layer_tag(self, idx: int) -> str:
        """Generate logging tag for a layer."""
        return f"layer{idx}"

    def train(
        self,
        epochs: int,
        log_every_pca: int = 25,
        log_every_probe: int = 10
    ):
        """
        Train iDBN with layer-wise greedy pretraining.

        Logs reconstruction quality, PCA embeddings, and linear probe performance.

        Args:
            epochs: Number of training epochs
            log_every_pca: Log PCA embeddings every N epochs
            log_every_probe: Log linear probe performance every N epochs
        """
        for epoch in tqdm(range(int(epochs)), desc="iDBN"):
            losses = []

            # Forward pass through all layers
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.layers:
                    loss = rbm.train_epoch(v, epoch, epochs, CD=self.cd_k)
                    v = rbm.forward(v)
                    losses.append(float(loss))

            # Log training loss
            if self.wandb_run and losses:
                self.wandb_run.log({"idbn/loss": float(np.mean(losses)), "epoch": epoch})

            # Auto-reconstruction snapshot every 5 epochs
            if self.wandb_run and self.val_batch is not None and epoch % 5 == 0:
                with torch.no_grad():
                    rec = self.reconstruct(self.val_batch[:8].to(self.device))
                img0 = self.val_batch[:8]
                try:
                    B, C, H, W = img0.shape
                    recv = rec.view(B, C, H, W).clamp(0, 1)
                except Exception:
                    side = int(rec.size(1) ** 0.5)
                    C = 1
                    H = W = side
                    recv = rec.view(-1, C, H, W).clamp(0, 1)
                    img0 = img0.view(-1, C, H, W)

                grid = vutils.make_grid(torch.cat([img0.cpu(), recv.cpu()], dim=0), nrow=img0.size(0))
                self.wandb_run.log({
                    "idbn/auto_recon_grid": wandb.Image(grid.permute(1, 2, 0).numpy()),
                    "epoch": epoch
                })

                try:
                    mse = F.mse_loss(
                        img0.to(self.device).view(img0.size(0), -1),
                        recv.view(img0.size(0), -1)
                    )
                    self.wandb_run.log({"idbn/auto_recon_mse": mse.item(), "epoch": epoch})
                except Exception:
                    pass

            # PCA visualization and linear probes for monitored layers
            if self.wandb_run and self.val_loader is not None and self.features is not None:

                # Log PCA embeddings
                if epoch % log_every_pca == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            E, feats = compute_val_embeddings_and_features(self, upto_layer=layer_idx)
                            if E.numel() == 0:
                                continue

                            emb_np = E.numpy()
                            feat_map = {
                                "Cumulative Area": feats["cum_area"].numpy(),
                                "Convex Hull": feats["convex_hull"].numpy(),
                                "Labels": feats["labels"].numpy(),
                            }
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()

                            # 2D PCA plot
                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2,
                                    features=feat_map,
                                    arch_name=f"iDBN_{tag}",
                                    dist_name="val",
                                    method_name="pca",
                                    wandb_run=self.wandb_run,
                                )

                                # 3D PCA plot if sufficient dimensions
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3,
                                        features=feat_map,
                                        arch_name=f"iDBN_{tag}",
                                        dist_name="val",
                                        method_name="pca",
                                        wandb_run=self.wandb_run,
                                    )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_pca_error_{tag}": str(e)})

                # Log linear probe performance
                if epoch % log_every_probe == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            log_linear_probe(
                                self,
                                epoch=epoch,
                                n_bins=5,
                                test_size=0.2,
                                steps=1000,
                                lr=1e-2,
                                patience=20,
                                min_delta=0.0,
                                upto_layer=layer_idx,
                                layer_tag=tag,
                            )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_probe_error_{tag}": str(e)})

    @torch.no_grad()
    def represent(self, x: torch.Tensor, upto_layer: Optional[int] = None) -> torch.Tensor:
        """
        Compute representation at a given layer.

        Args:
            x: Input tensor (images, shape [B, C, H, W] or [B, D])
            upto_layer: Which layer to compute to (None = all layers)

        Returns:
            Activations at specified layer [B, hidden_size]
        """
        v = x.view(x.size(0), -1).float().to(self.device)
        L = len(self.layers) if (upto_layer is None) else max(0, min(len(self.layers), int(upto_layer)))
        for i in range(L):
            v = self.layers[i].forward(v)
        return v

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input by encoding through all layers then decoding back.

        Args:
            x: Input tensor [B, C, H, W] or [B, D]

        Returns:
            Reconstructed tensor [B, D]
        """
        v = x.view(x.size(0), -1).float().to(self.device)
        # Encode: forward through all layers
        cur = v
        for rbm in self.layers:
            cur = rbm.forward(cur)
        # Decode: backward through layers in reverse order
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def decode(self, top: torch.Tensor) -> torch.Tensor:
        """
        Decode from top layer activations back to input space.

        Args:
            top: Top layer activations [B, top_hidden_size]

        Returns:
            Reconstructed input [B, input_size]
        """
        cur = top.to(self.device)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def save_model(self, path: str):
        """
        Save iDBN model to disk.

        Saves the RBM layers and training parameters for later loading.

        Args:
            path: Output file path (.pkl)
        """
        model_copy = {"layers": self.layers, "params": self.params}
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)
        print(f"[iDBN] Model saved to {path}")
