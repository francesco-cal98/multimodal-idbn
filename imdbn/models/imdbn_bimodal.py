"""
iMDBN_BiModal (Image-Image Multimodal Deep Belief Network)

A bimodal extension that jointly models two visual modalities:
- Numerosity images (dots/objects)
- MNIST-100 images (handwritten numbers)

Both modalities are processed by separate DBNs and combined via a joint RBM.
"""

import os
import pickle
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from sklearn.decomposition import PCA

from imdbn.models.idbn import iDBN
from imdbn.models.rbm import RBM
from imdbn.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
from imdbn.utils.probe_utils import (
    log_linear_probe,
    compute_val_embeddings_and_features,
    train_linear_classifier,
    make_bin_labels,
    stratified_split,
    _format_bin_names,
    _confusion_df,
    _log_accuracy_wandb,
    _log_confusion_table_wandb,
)


def compute_bimodal_joint_embeddings_and_features(model):
    """Compute joint embeddings and features for bimodal model."""
    all_embeddings = []

    for mod1, mod2 in model.val_loader:
        v1 = mod1.to(model.device).view(mod1.size(0), -1).float()
        v2 = mod2.to(model.device).view(mod2.size(0), -1).float()

        z1 = model.mod1_dbn.represent(v1)
        z2 = model.mod2_dbn.represent(v2)

        # Pass through all joint layers
        h = torch.cat([z1, z2], dim=1)
        for rbm in model.joint_layers:
            h = rbm.forward(h)
        all_embeddings.append(h.cpu())

    E = torch.cat(all_embeddings, dim=0)

    feats = {}
    if model.features is not None:
        if "Cumulative Area" in model.features:
            feats["cum_area"] = model.features["Cumulative Area"]
        if "Convex Hull" in model.features:
            feats["convex_hull"] = model.features["Convex Hull"]
        if "Labels" in model.features:
            feats["labels"] = model.features["Labels"]
        if "Density" in model.features:
            feats["density"] = model.features["Density"]

    return E, feats


def log_bimodal_joint_linear_probe(model, epoch, n_bins=5, test_size=0.2,
                                    steps=1000, lr=1e-2, patience=20, min_delta=0.0,
                                    metric_prefix="joint", save_csv=False):
    """
    Log linear probes for bimodal joint representation.

    Probes: cum_area, convex_hull, labels, density (if available)
    All features are binned into n_bins for classification.
    """
    import matplotlib.pyplot as plt

    E, feats = compute_bimodal_joint_embeddings_and_features(model)
    if E.numel() == 0:
        return
    E_np = E.detach().numpy()

    probe_targets = ["cum_area", "convex_hull", "labels"]
    if "density" in feats:
        probe_targets.append("density")

    summary_rows = []

    for mkey in probe_targets:
        if mkey not in feats:
            continue

        # Bin the feature
        vals = feats[mkey].to(torch.float32)
        y, edges = make_bin_labels(vals, n_bins=n_bins)
        bin_names = _format_bin_names(edges, precision=4)
        metric_name = f"{metric_prefix}/{mkey}"

        # Stratified split
        train_idx, test_idx = stratified_split(y, test_size=test_size, rng_seed=42)
        if len(train_idx) == 0 or len(test_idx) == 0:
            if model.wandb_run:
                model.wandb_run.log({f"{metric_name}/warn_empty_split": 0.0, "epoch": epoch})
            continue

        # Train and evaluate
        Xtr, ytr = E_np[train_idx], y.numpy()[train_idx]
        Xte, yte = E_np[test_idx], y.numpy()[test_idx]

        acc, y_true, y_pred = train_linear_classifier(
            Xtr, ytr, Xte, yte,
            device=model.device,
            n_classes=n_bins,
            max_steps=steps,
            lr=lr,
            weight_decay=0.0,
            patience=patience,
            min_delta=min_delta,
        )

        summary_rows.append((metric_name, acc))

        # Confusion matrix
        df = _confusion_df(y_true, y_pred, n_bins, bin_names)

        # Log to W&B
        _log_accuracy_wandb(model.wandb_run, metric_name, acc, epoch)
        _log_confusion_table_wandb(model.wandb_run, df, metric_name, epoch)

    # Summary bar plot
    if summary_rows and model.wandb_run:
        labels_plot = [name for name, _ in summary_rows]
        values = [val for _, val in summary_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels_plot) * 1.2), 4))
        ax.bar(range(len(labels_plot)), values, color='indianred')
        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Joint probe summary @ epoch {epoch}")
        fig.tight_layout()
        model.wandb_run.log({f"probe/{metric_prefix}/summary": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)


@torch.no_grad()
def log_bimodal_latent_trajectory(
    model,
    sample_idx: int = 0,
    steps: int = 50,
    tag: str = "trajectory",
    n_frames: int = 8,
):
    """
    Log latent trajectory visualization for bimodal model.

    Shows trajectory from MOD2 -> MOD1 in the joint latent space,
    with reconstruction panels.
    """
    import matplotlib.pyplot as plt

    if model.val_loader is None or model.wandb_run is None:
        return

    # Collect all joint embeddings for PCA background
    all_z = []
    all_labels = []
    for mod1, mod2 in model.val_loader:
        v1 = mod1.to(model.device).view(mod1.size(0), -1).float()
        v2 = mod2.to(model.device).view(mod2.size(0), -1).float()
        z1 = model.mod1_dbn.represent(v1)
        z2 = model.mod2_dbn.represent(v2)
        z_concat = torch.cat([z1, z2], dim=1)
        all_z.append(z_concat.cpu())
        # Get labels if available
        try:
            base = model.val_loader.dataset.dataset
            indices = model.val_loader.dataset.indices
            batch_indices = indices[len(all_labels):len(all_labels) + mod1.size(0)]
            batch_labels = [base.labels[i] for i in batch_indices]
            all_labels.extend(batch_labels)
        except Exception:
            all_labels.extend([0] * mod1.size(0))

    Z_all = torch.cat(all_z, dim=0).numpy()
    color_vec = np.array(all_labels, dtype=float)
    N_val = Z_all.shape[0]
    sample_idx = min(sample_idx, N_val - 1)

    # Get the sample
    seen = 0
    mod1_sample = None
    mod2_sample = None
    for mod1, mod2 in model.val_loader:
        b = mod1.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        mod1_sample = mod1[pos:pos+1].to(model.device)
        mod2_sample = mod2[pos:pos+1].to(model.device)
        break

    if mod1_sample is None:
        return

    # Compute initial latents
    v1 = mod1_sample.view(1, -1).float()
    v2 = mod2_sample.view(1, -1).float()
    z1_true = model.mod1_dbn.represent(v1)
    z2_true = model.mod2_dbn.represent(v2)

    Dz1, Dz2 = model.Dz_mod1, model.Dz_mod2
    V = Dz1 + Dz2

    # MOD2 -> MOD1 trajectory (given MNIST, infer numerosity)
    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz1:] = z2_true
    km[:, Dz1:] = 1.0

    # Initialize MOD1 from random
    h0 = model.joint_rbm.forward(v_known)
    v_prob0 = model.joint_rbm.visible_probs(h0)
    v_cur = v_prob0 * (1 - km) + v_known * km

    # Collect trajectory in h_joint space (hidden of joint RBM)
    traj_h = [h0.detach().cpu().numpy()]
    recon_imgs = []

    # Initial reconstruction
    img0 = model.mod1_dbn.decode(v_cur[:, :Dz1]).detach().cpu()
    recon_imgs.append(img0)

    # Run Gibbs sampling
    for step_i in range(steps):
        h_prob = model.joint_rbm.forward(v_cur)
        h_sample = torch.bernoulli(h_prob)
        v_logits = h_sample @ model.joint_rbm.W.T + model.joint_rbm.vis_bias
        v_prob = torch.sigmoid(v_logits)
        v_cur = v_prob * (1 - km) + v_known * km

        traj_h.append(h_prob.detach().cpu().numpy())  # Track h_joint
        img_t = model.mod1_dbn.decode(v_cur[:, :Dz1]).detach().cpu()
        recon_imgs.append(img_t)

    traj_h = np.vstack(traj_h)  # [steps+1, H_joint]

    # Compute h_joint for all validation samples
    all_h_joint = []
    for mod1, mod2 in model.val_loader:
        v1_batch = mod1.to(model.device).view(mod1.size(0), -1).float()
        v2_batch = mod2.to(model.device).view(mod2.size(0), -1).float()
        z1_batch = model.mod1_dbn.represent(v1_batch)
        z2_batch = model.mod2_dbn.represent(v2_batch)
        v_joint = torch.cat([z1_batch, z2_batch], dim=1)
        h_joint = model.joint_rbm.forward(v_joint)
        all_h_joint.append(h_joint.cpu())
    H_all = torch.cat(all_h_joint, dim=0).numpy()

    # PCA for visualization (fit on all h_joint from validation)
    pca = PCA(n_components=2)
    H2d = pca.fit_transform(H_all)
    traj_2d = pca.transform(traj_h)

    # Get h_joint for the true sample
    v_true_joint = torch.cat([z1_true, z2_true], dim=1)
    h_true = model.joint_rbm.forward(v_true_joint)
    h_true_2d = pca.transform(h_true.cpu().numpy())

    # Select frames for panel
    sel_idx = np.unique(np.linspace(0, len(recon_imgs) - 1, n_frames, dtype=int)).tolist()

    # Helper to convert to image
    Npix = v1.size(1)
    side = int(round(Npix ** 0.5))
    def _to_img(t):
        return t.view(-1).clamp(0, 1).view(side, side).numpy()

    # Create figure
    import math
    n_tiles = len(sel_idx) + 1  # GT + selected frames
    rows = 2
    cols = math.ceil(n_tiles / rows)

    fig = plt.figure(figsize=(8 + cols * 2.2, max(6, rows * 2.2)))
    gs = fig.add_gridspec(nrows=rows, ncols=cols + 4)

    # PCA scatter plot in h_joint space
    ax0 = fig.add_subplot(gs[:, :4])
    sc = ax0.scatter(H2d[:, 0], H2d[:, 1], c=color_vec, cmap="viridis", s=12, alpha=0.35)
    ax0.scatter(h_true_2d[0, 0], h_true_2d[0, 1], s=80, marker="*", c="k",
                edgecolor="w", linewidths=0.8, label="GT", zorder=3)
    ax0.scatter(traj_2d[0, 0], traj_2d[0, 1], s=50, marker="D", c="red",
                edgecolor="k", linewidths=0.5, label="start", zorder=3)
    ax0.plot(traj_2d[:, 0], traj_2d[:, 1], linewidth=1.6, marker="o",
             markersize=3, c="red", label="trajectory", zorder=2)

    ax0.set_title(f"PCA h_joint — sample {sample_idx} — steps={steps}")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02, label="Numerosity")
    ax0.legend(loc="best")

    # Reconstruction panels
    right_gs = gs[:, 4:].subgridspec(nrows=rows, ncols=cols)

    # GT image
    ax = fig.add_subplot(right_gs[0, 0])
    ax.imshow(_to_img(v1.cpu()), cmap="gray", vmin=0, vmax=1)
    ax.set_title("GT", fontsize=9)
    ax.axis("off")

    # Selected frames
    for k, si in enumerate(sel_idx):
        r = (k + 1) // cols
        c = (k + 1) % cols
        ax = fig.add_subplot(right_gs[r, c])
        ax.imshow(_to_img(recon_imgs[si]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"step {si}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    model.wandb_run.log({f"{tag}/mod2_to_mod1": wandb.Image(fig)})
    plt.close(fig)


@torch.no_grad()
def log_bimodal_latent_trajectory_3d(
    model,
    sample_idx: int = 0,
    steps: int = 50,
    tag: str = "trajectory",
):
    """Log 3D latent trajectory visualization for bimodal model."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if model.val_loader is None or model.wandb_run is None:
        return

    # Collect embeddings
    all_z1 = []
    for mod1, mod2 in model.val_loader:
        v1 = mod1.to(model.device).view(mod1.size(0), -1).float()
        z1 = model.mod1_dbn.represent(v1)
        all_z1.append(z1.cpu())

    Z1_all = torch.cat(all_z1, dim=0).numpy()
    N_val = Z1_all.shape[0]
    sample_idx = min(sample_idx, N_val - 1)

    # Get sample
    seen = 0
    mod2_sample = None
    z1_true = None
    for mod1, mod2 in model.val_loader:
        b = mod1.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        mod1_sample = mod1[pos:pos+1].to(model.device)
        mod2_sample = mod2[pos:pos+1].to(model.device)
        z1_true = model.mod1_dbn.represent(mod1_sample.view(1, -1).float())
        break

    if mod2_sample is None:
        return

    z2 = model.mod2_dbn.represent(mod2_sample.view(1, -1).float())

    Dz1, Dz2 = model.Dz_mod1, model.Dz_mod2
    V = Dz1 + Dz2

    # MOD2 -> MOD1 trajectory
    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz1:] = z2
    km[:, Dz1:] = 1.0

    h0 = model.joint_rbm.forward(v_known)
    v_prob0 = model.joint_rbm.visible_probs(h0)
    v_cur = v_prob0 * (1 - km) + v_known * km

    traj_z = [v_cur[:, :Dz1].detach().cpu().numpy()]
    for _ in range(steps):
        h_prob = model.joint_rbm.forward(v_cur)
        h_sample = torch.bernoulli(h_prob)
        v_logits = h_sample @ model.joint_rbm.W.T + model.joint_rbm.vis_bias
        v_prob = torch.sigmoid(v_logits)
        v_cur = v_prob * (1 - km) + v_known * km
        traj_z.append(v_cur[:, :Dz1].detach().cpu().numpy())

    traj_z = np.vstack(traj_z)

    # PCA-3
    pca3 = PCA(n_components=3)
    Z3 = pca3.fit_transform(Z1_all)
    T3 = pca3.transform(traj_z)

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=6, alpha=0.15)
    ax.plot(T3[:, 0], T3[:, 1], T3[:, 2], c="r", linewidth=1.2)
    ax.set_title(f"PCA-3 trajectory sample {sample_idx}")
    fig.tight_layout()

    model.wandb_run.log({f"{tag}/mod2_to_mod1_3d": wandb.Image(fig)})
    plt.close(fig)


class iMDBN_BiModal(nn.Module):
    """
    Bimodal Deep Belief Network for two visual modalities.

    Architecture:
    - Modality 1 (Numerosity): DBN [10000 -> 1500 -> 500]
    - Modality 2 (MNIST-100): DBN [1568 -> 500 -> 500]
    - Joint RBM: [500 + 500 -> joint_hidden]

    Supports:
    - Independent pre-training of each modality DBN
    - Joint training with cross-modal reconstruction
    - Bidirectional inference (numerosity <-> MNIST-100)
    """

    def __init__(
        self,
        layer_sizes_mod1: list,
        layer_sizes_mod2: list,
        joint_layer_sizes: list,
        params: Optional[dict] = None,
        dataloader=None,
        val_loader=None,
        device=None,
        wandb_run=None,
        logging_cfg: Optional[dict] = None,
    ):
        """
        Initialize BiModal DBN.

        Args:
            layer_sizes_mod1: Modality 1 (numerosity) DBN layers, e.g., [10000, 1500, 500]
            layer_sizes_mod2: Modality 2 (MNIST-100) DBN layers, e.g., [1568, 500, 500]
            joint_layer_sizes: Joint DBN hidden layer sizes, e.g., [1500] or [1500, 1500]
            params: Training hyperparameters
            dataloader: Training DataLoader yielding (mod1_images, mod2_images)
            val_loader: Validation DataLoader
            device: torch.device
            wandb_run: Optional W&B run for logging
            logging_cfg: Logging configuration dict
        """
        super().__init__()

        self.params = params or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        self.logging_cfg = logging_cfg or {}

        # Build Modality 1 DBN (Numerosity)
        self.mod1_dbn = iDBN(
            layer_sizes=layer_sizes_mod1,
            params=self.params,
            dataloader=None,  # Will be set during training
            val_loader=None,
            device=self.device,
            wandb_run=self.wandb_run,
        )

        # Build Modality 2 DBN (MNIST-100)
        self.mod2_dbn = iDBN(
            layer_sizes=layer_sizes_mod2,
            params=self.params,
            dataloader=None,
            val_loader=None,
            device=self.device,
            wandb_run=self.wandb_run,
        )

        # Latent dimensions
        self.Dz_mod1 = int(self.mod1_dbn.layers[-1].num_hidden)
        self.Dz_mod2 = int(self.mod2_dbn.layers[-1].num_hidden)

        # Build joint DBN (can be multi-layer)
        self._build_joint(joint_layer_sizes)

        # Training hyperparameters
        self.joint_cd = int(self.params.get("JOINT_CD", self.params.get("CD", 1)))
        self.cross_steps = int(self.params.get("CROSS_GIBBS_STEPS", 50))

        # Cache validation batch
        try:
            vb_mod1, vb_mod2 = next(iter(val_loader))
            self.validation_mod1 = vb_mod1[:8].to(self.device)
            self.validation_mod2 = vb_mod2[:8].to(self.device)
        except Exception:
            self.validation_mod1 = None
            self.validation_mod2 = None

        # Extract validation features for probes
        self.features = None
        try:
            # Handle both Subset and direct dataset
            if hasattr(val_loader.dataset, 'indices'):
                indices = val_loader.dataset.indices
                base = val_loader.dataset.dataset
            else:
                base = val_loader.dataset
                indices = range(len(base))

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
            print(f"[iMDBN_BiModal] Extracted features: {list(self.features.keys())}")
        except Exception as e:
            print(f"[iMDBN_BiModal] Warning: Could not extract features for probes: {e}")

        # Build architecture string (handle both int and list for joint_layer_sizes)
        joint_sizes_for_str = joint_layer_sizes if isinstance(joint_layer_sizes, list) else [joint_layer_sizes]
        self.arch_str = f"MOD1{'-'.join(map(str, layer_sizes_mod1))}_MOD2{'-'.join(map(str, layer_sizes_mod2))}_JOINT{'-'.join(map(str, joint_sizes_for_str))}"

    def _build_joint(self, joint_layer_sizes: list):
        """
        Build joint DBN connecting both modality latents.

        Args:
            joint_layer_sizes: List of hidden layer sizes, e.g., [1500] or [1500, 1500]
        """
        # Convert to list if single integer was passed
        if isinstance(joint_layer_sizes, int):
            joint_layer_sizes = [joint_layer_sizes]

        total_visible = self.Dz_mod1 + self.Dz_mod2
        self.joint_layers = []

        # Build RBM stack for joint DBN
        current_visible = total_visible
        for i, hidden_size in enumerate(joint_layer_sizes):
            rbm = RBM(
                num_visible=current_visible,
                num_hidden=int(hidden_size),
                learning_rate=self.params.get("JOINT_LEARNING_RATE", self.params.get("LEARNING_RATE", 0.1)),
                weight_decay=self.params.get("WEIGHT_PENALTY", 0.0001),
                momentum=self.params.get("INIT_MOMENTUM", 0.5),
                dynamic_lr=self.params.get("LEARNING_RATE_DYNAMIC", True),
                final_momentum=self.params.get("FINAL_MOMENTUM", 0.95),
                softmax_groups=[],  # No softmax for image-image
            ).to(self.device)
            self.joint_layers.append(rbm)
            current_visible = int(hidden_size)

        # Keep backward compatibility: joint_rbm points to the first layer
        self.joint_rbm = self.joint_layers[0]
        self.num_joint_layers = len(self.joint_layers)

    def load_pretrained_mod1_dbn(self, path: str) -> bool:
        """Load pretrained Modality 1 (numerosity) DBN."""
        return self._load_pretrained_dbn(self.mod1_dbn, path, "mod1")

    def load_pretrained_mod2_dbn(self, path: str) -> bool:
        """Load pretrained Modality 2 (MNIST-100) DBN."""
        return self._load_pretrained_dbn(self.mod2_dbn, path, "mod2")

    def _load_pretrained_dbn(self, dbn: iDBN, path: str, name: str) -> bool:
        """Generic loader for pretrained DBN."""
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[load_pretrained_{name}_dbn] error: {e}")
            return False

        if isinstance(obj, dict) and "layers" in obj:
            dbn.layers = obj["layers"]
        elif hasattr(obj, "layers"):
            dbn.layers = obj.layers
        else:
            print(f"[load_pretrained_{name}_dbn] unrecognized format")
            return False

        # Move to device
        for rbm in dbn.layers:
            rbm.W = rbm.W.to(self.device)
            rbm.hid_bias = rbm.hid_bias.to(self.device)
            rbm.vis_bias = rbm.vis_bias.to(self.device)
            rbm.W_m = torch.zeros_like(rbm.W)
            rbm.hb_m = torch.zeros_like(rbm.hid_bias)
            rbm.vb_m = torch.zeros_like(rbm.vis_bias)
            if not hasattr(rbm, "softmax_groups"):
                rbm.softmax_groups = []

        print(f"[load_pretrained_{name}_dbn] loaded from {path}")
        return True

    @torch.no_grad()
    def init_joint_bias_from_data(self, n_batches: int = 10):
        """Initialize joint DBN biases from data statistics (first layer only)."""
        sum_z1 = None
        sum_z2 = None
        n = 0

        for b, (mod1, mod2) in enumerate(self.dataloader):
            if b >= n_batches:
                break

            v1 = mod1.to(self.device).view(mod1.size(0), -1).float()
            v2 = mod2.to(self.device).view(mod2.size(0), -1).float()

            z1 = self.mod1_dbn.represent(v1)
            z2 = self.mod2_dbn.represent(v2)

            sum_z1 = z1.sum(0) if sum_z1 is None else (sum_z1 + z1.sum(0))
            sum_z2 = z2.sum(0) if sum_z2 is None else (sum_z2 + z2.sum(0))
            n += z1.size(0)

        if n == 0:
            return

        mean_z1 = (sum_z1 / n).clamp(1e-4, 1 - 1e-4)
        mean_z2 = (sum_z2 / n).clamp(1e-4, 1 - 1e-4)

        # Set biases for the first joint layer
        self.joint_layers[0].vis_bias[:self.Dz_mod1] = torch.log(mean_z1) - torch.log1p(-mean_z1)
        self.joint_layers[0].vis_bias[self.Dz_mod1:] = torch.log(mean_z2) - torch.log1p(-mean_z2)

    @torch.no_grad()
    def _cross_reconstruct(
        self,
        z_mod1: torch.Tensor,
        z_mod2: torch.Tensor,
        steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal reconstruction via joint RBM.

        Returns:
            Tuple of (mod1_from_mod2, mod2_from_mod1)
        """
        if steps is None:
            steps = self.cross_steps

        B = z_mod1.size(0)
        Dz1, Dz2 = self.Dz_mod1, self.Dz_mod2
        V = Dz1 + Dz2

        # MOD1 -> MOD2: Given numerosity latents, infer MNIST-100 latents
        v_known = torch.zeros(B, V, device=self.device)
        km = torch.zeros_like(v_known)
        v_known[:, :Dz1] = z_mod1
        km[:, :Dz1] = 1.0

        v_mod1_to_mod2 = self.joint_rbm.conditional_gibbs(
            v_known, km, n_steps=steps, sample_h=True, sample_v=False
        )
        z_mod2_from_mod1 = v_mod1_to_mod2[:, Dz1:]

        # MOD2 -> MOD1: Given MNIST-100 latents, infer numerosity latents
        v_known.zero_()
        km.zero_()
        v_known[:, Dz1:] = z_mod2
        km[:, Dz1:] = 1.0

        v_mod2_to_mod1 = self.joint_rbm.conditional_gibbs(
            v_known, km, n_steps=steps, sample_h=True, sample_v=False
        )
        z_mod1_from_mod2 = v_mod2_to_mod1[:, :Dz1]

        # Decode to image space
        mod1_from_mod2 = self.mod1_dbn.decode(z_mod1_from_mod2)
        mod2_from_mod1 = self.mod2_dbn.decode(z_mod2_from_mod1)

        return mod1_from_mod2, mod2_from_mod1

    @torch.no_grad()
    def represent(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute joint representation from both modalities through all joint layers."""
        mod1_data, mod2_data = batch
        v1 = mod1_data.to(self.device).view(mod1_data.size(0), -1).float()
        v2 = mod2_data.to(self.device).view(mod2_data.size(0), -1).float()

        z1 = self.mod1_dbn.represent(v1)
        z2 = self.mod2_dbn.represent(v2)

        # Pass through all joint layers
        h = torch.cat([z1, z2], dim=1)
        for rbm in self.joint_layers:
            h = rbm.forward(h)
        return h

    def train_joint(
        self,
        epochs: int,
        log_every: int = 5,
        log_every_pca: int = 25,
        log_every_probe: int = 10,
        log_every_trajectory: int = 50,
    ):
        """
        Train joint DBN connecting both modalities with iterative training of all layers.

        Training strategy:
        - Warmup (first 8 epochs): Alternating modality clamping (first layer only)
        - Main (remaining epochs): Free CD + auxiliary clamped training (all layers)

        Logged metrics:
        - Cross-modal: MSE for both modalities
        - Embeddings: PCA projections of joint latents
        - Linear probes: Downstream task performance
        """
        print(f"[iMDBN_BiModal] joint training: {self.num_joint_layers} layers, {epochs} epochs total")
        self.init_joint_bias_from_data(n_batches=10)

        WARMUP_EPOCHS = 8
        Dz1, Dz2 = self.Dz_mod1, self.Dz_mod2
        V = Dz1 + Dz2
        aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 30))

        for epoch in tqdm(range(int(epochs)), desc="Joint"):
            cd_losses = []
            totals = {"n": 0, "mse_mod1": 0.0, "mse_mod2": 0.0}

            for b_idx, (mod1, mod2) in enumerate(self.dataloader):
                v1 = mod1.to(self.device).view(mod1.size(0), -1).float()
                v2 = mod2.to(self.device).view(mod2.size(0), -1).float()
                B = v1.size(0)

                with torch.no_grad():
                    z1 = self.mod1_dbn.represent(v1)
                    z2 = self.mod2_dbn.represent(v2)
                    v_plus = torch.cat([z1, z2], dim=1)

                if epoch < WARMUP_EPOCHS:
                    # Warmup: alternating modality clamping, 2x per batch (first layer only)
                    for _ in range(2):
                        # Clamp mod1
                        v_known = torch.zeros(B, V, device=self.device)
                        km = torch.zeros(B, V, device=self.device)
                        v_known[:, :Dz1] = z1
                        km[:, :Dz1] = 1.0
                        self.joint_layers[0].train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=3, cond_init_steps=aux_cond_steps,
                            sample_h=True, sample_v=False,
                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )

                        # Clamp mod2
                        v_known.zero_()
                        km.zero_()
                        v_known[:, Dz1:] = z2
                        km[:, Dz1:] = 1.0
                        self.joint_layers[0].train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=3, cond_init_steps=aux_cond_steps,
                            sample_h=True, sample_v=False,
                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )
                else:
                    # Main training: train all layers iteratively
                    # Layer 0: train with concatenated latents
                    current_input = v_plus
                    for layer_idx, rbm in enumerate(self.joint_layers):
                        loss_cd = rbm.train_epoch(current_input, epoch, epochs, CD=self.joint_cd)
                        if layer_idx == 0:
                            cd_losses.append(float(loss_cd))

                        # Get output for next layer (with no_grad since we're going layer by layer)
                        with torch.no_grad():
                            current_input = rbm.forward(current_input)

                    # Auxiliary clamped training - Modality 1 (first layer only)
                    v_known = torch.zeros(B, V, device=self.device)
                    km = torch.zeros(B, V, device=self.device)
                    v_known[:, :Dz1] = z1
                    km[:, :Dz1] = 1.0
                    self.joint_layers[0].train_epoch_clamped(
                        v_known, km, epoch, epochs,
                        CD=3, cond_init_steps=aux_cond_steps,
                        sample_h=True, sample_v=False,
                        reclamp_negative=False,
                        aux_lr_mult=0.3,
                        use_noisy_init=True,
                    )

                    # Auxiliary clamped training - Modality 2 (first layer only)
                    v_known.zero_()
                    km.zero_()
                    v_known[:, Dz1:] = z2
                    km[:, Dz1:] = 1.0
                    self.joint_layers[0].train_epoch_clamped(
                        v_known, km, epoch, epochs,
                        CD=3, cond_init_steps=aux_cond_steps,
                        sample_h=True, sample_v=False,
                        reclamp_negative=False,
                        aux_lr_mult=0.3,
                        use_noisy_init=True,
                    )

                # Cross-modal reconstruction metrics
                with torch.no_grad():
                    mod1_rec, mod2_rec = self._cross_reconstruct(z1, z2, steps=self.cross_steps)

                    mse1 = F.mse_loss(mod1_rec.view_as(v1), v1, reduction="sum")
                    mse2 = F.mse_loss(mod2_rec.view_as(v2), v2, reduction="sum")

                    totals["n"] += B
                    totals["mse_mod1"] += float(mse1.item())
                    totals["mse_mod2"] += float(mse2.item())

            # Log training metrics
            if self.wandb_run and cd_losses:
                self.wandb_run.log({
                    "joint/cd_loss": float(np.mean(cd_losses)),
                    "epoch": epoch
                })

            if self.wandb_run and totals["n"] > 0:
                npix1 = self.mod1_dbn.layers[0].num_visible
                npix2 = self.mod2_dbn.layers[0].num_visible
                mse1_mean = totals["mse_mod1"] / (totals["n"] * npix1)
                mse2_mean = totals["mse_mod2"] / (totals["n"] * npix2)

                self.wandb_run.log({
                    "cross_modality/mod1_mse": mse1_mean,
                    "cross_modality/mod2_mse": mse2_mean,
                    "epoch": epoch
                })

            # Visual logging and probes
            if self.wandb_run and self.val_loader is not None:

                # PCA embeddings
                if epoch % log_every_pca == 0:
                    try:
                        # Joint embeddings PCA
                        E, feats = compute_bimodal_joint_embeddings_and_features(self)
                        if E.numel() > 0:
                            emb_np = E.detach().numpy()
                            feat_map = {}
                            if "cum_area" in feats:
                                feat_map["Cumulative Area"] = feats["cum_area"].numpy()
                            if "convex_hull" in feats:
                                feat_map["Convex Hull"] = feats["convex_hull"].numpy()
                            if "labels" in feats:
                                feat_map["Labels"] = feats["labels"].numpy()
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()

                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2, features=feat_map,
                                    arch_name="Joint_bimodal", dist_name="val",
                                    method_name="pca", wandb_run=self.wandb_run,
                                )
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3, features=feat_map,
                                        arch_name="Joint_bimodal", dist_name="val",
                                        method_name="pca", wandb_run=self.wandb_run,
                                    )

                        # MOD2 (MNIST-100) latent PCA
                        all_z2 = []
                        for mod1, mod2 in self.val_loader:
                            v2 = mod2.to(self.device).view(mod2.size(0), -1).float()
                            z2 = self.mod2_dbn.represent(v2)
                            all_z2.append(z2.cpu())
                        Z2_all = torch.cat(all_z2, dim=0).numpy()

                        if Z2_all.shape[0] > 2 and Z2_all.shape[1] > 2:
                            p2_mod2 = PCA(n_components=2).fit_transform(Z2_all)
                            # For MNIST-100 PCA, only use Labels (numerosity value) for coloring
                            mnist_feat_map = {}
                            if "labels" in feats:
                                mnist_feat_map["Labels"] = feats["labels"].numpy()
                            plot_2d_embedding_and_correlations(
                                emb_2d=p2_mod2, features=mnist_feat_map,
                                arch_name="MOD2_MNIST100", dist_name="val",
                                method_name="pca", wandb_run=self.wandb_run,
                            )
                            if Z2_all.shape[1] >= 3:
                                p3_mod2 = PCA(n_components=3).fit_transform(Z2_all)
                                plot_3d_embedding_and_correlations(
                                    emb_3d=p3_mod2, features=mnist_feat_map,
                                    arch_name="MOD2_MNIST100", dist_name="val",
                                    method_name="pca", wandb_run=self.wandb_run,
                                )

                    except Exception as e:
                        print(f"[PCA ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                        self.wandb_run.log({"warn/joint_pca_error": str(e)})

                # Linear probes
                if epoch % log_every_probe == 0:
                    try:
                        log_bimodal_joint_linear_probe(
                            self,
                            epoch=epoch,
                            n_bins=5, test_size=0.2,
                            steps=1000, lr=1e-2, patience=20, min_delta=0.0,
                            metric_prefix="joint",
                        )
                    except Exception as e:
                        self.wandb_run.log({"warn/joint_probe_error": str(e)})

                # Latent trajectory visualization
                if epoch % log_every_trajectory == 0:
                    traj_cfg = self.logging_cfg.get("trajectory", {})
                    if traj_cfg.get("enable", False):
                        try:
                            num_samples = traj_cfg.get("num_samples", 4)
                            for s_idx in range(num_samples):
                                log_bimodal_latent_trajectory(
                                    self,
                                    sample_idx=s_idx,
                                    steps=self.cross_steps,
                                    tag=f"trajectory/sample{s_idx}",
                                    n_frames=8,
                                )
                            if traj_cfg.get("plot_3d", True):
                                log_bimodal_latent_trajectory_3d(
                                    self,
                                    sample_idx=0,
                                    steps=self.cross_steps,
                                    tag="trajectory",
                                )
                        except Exception as e:
                            self.wandb_run.log({"warn/trajectory_error": str(e)})

            # Snapshots
            if epoch % max(1, int(log_every)) == 0:
                self._log_snapshots(epoch)

        print("[iMDBN_BiModal] joint training finished.")

    @torch.no_grad()
    def _log_snapshots(self, epoch: int, num: int = 8):
        """Log reconstruction snapshots."""
        if self.wandb_run is None or self.validation_mod1 is None:
            return

        mod1 = self.validation_mod1[:num]
        mod2 = self.validation_mod2[:num]

        z1 = self.mod1_dbn.represent(mod1.view(mod1.size(0), -1))
        z2 = self.mod2_dbn.represent(mod2.view(mod2.size(0), -1))

        mod1_rec, mod2_rec = self._cross_reconstruct(z1, z2, steps=self.cross_steps)

        # Log Modality 1 reconstruction (numerosity)
        B = mod1.size(0)
        side1 = int(round(mod1.view(B, -1).size(1) ** 0.5))
        mod1_4d = mod1.view(B, 1, side1, side1)
        mod1_rec_4d = mod1_rec.view(B, 1, side1, side1).clamp(0, 1)

        pair1 = torch.stack([mod1_4d.cpu(), mod1_rec_4d.cpu()], dim=1).view(-1, 1, side1, side1)
        grid1 = vutils.make_grid(pair1, nrow=2)
        self.wandb_run.log({
            "snap/mod1_from_mod2": wandb.Image(grid1.permute(1, 2, 0).numpy()),
            "epoch": epoch
        })

        # Log Modality 2 reconstruction (MNIST-100)
        # MNIST-100 is 28x56
        mod2_flat = mod2.view(B, -1)
        if mod2_flat.size(1) == 1568:  # 28x56
            mod2_4d = mod2_flat.view(B, 1, 28, 56)
            mod2_rec_4d = mod2_rec.view(B, 1, 28, 56).clamp(0, 1)
        else:
            side2 = int(round(mod2_flat.size(1) ** 0.5))
            mod2_4d = mod2_flat.view(B, 1, side2, side2)
            mod2_rec_4d = mod2_rec.view(B, 1, side2, side2).clamp(0, 1)

        pair2 = torch.stack([mod2_4d.cpu(), mod2_rec_4d.cpu()], dim=1).view(-1, 1, *mod2_4d.shape[2:])
        grid2 = vutils.make_grid(pair2, nrow=2)
        self.wandb_run.log({
            "snap/mod2_from_mod1": wandb.Image(grid2.permute(1, 2, 0).numpy()),
            "epoch": epoch
        })

        # MSE metrics
        mse1 = F.mse_loss(mod1_rec.view_as(mod1.view(B, -1)), mod1.view(B, -1)).item()
        mse2 = F.mse_loss(mod2_rec.view_as(mod2.view(B, -1)), mod2.view(B, -1)).item()
        self.wandb_run.log({
            "snap/mod1_mse": mse1,
            "snap/mod2_mse": mse2,
            "epoch": epoch
        })

    def save_model(self, path: str):
        """Save complete bimodal model."""
        import datetime

        payload = {
            "mod1_dbn": self.mod1_dbn,
            "mod2_dbn": self.mod2_dbn,
            "joint_layers": self.joint_layers,  # Save all joint layers
            "num_joint_layers": self.num_joint_layers,
            "Dz_mod1": self.Dz_mod1,
            "Dz_mod2": self.Dz_mod2,
            "params": self.params,
            "arch_str": self.arch_str,
            "features": self.features,
            "metadata": {
                "saved_at": datetime.datetime.now().isoformat(),
                "model_type": "iMDBN_BiModal",
                "architecture": self.arch_str,
            }
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"[iMDBN_BiModal] Model saved to {path}")
        print(f"[iMDBN_BiModal] Architecture: {self.arch_str}")

    @staticmethod
    def load_model(path: str, device=None) -> Dict[str, Any]:
        """Load bimodal model from disk."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Move to device
        if "mod1_dbn" in payload:
            for rbm in payload["mod1_dbn"].layers:
                rbm.to(device)

        if "mod2_dbn" in payload:
            for rbm in payload["mod2_dbn"].layers:
                rbm.to(device)

        # Support both old (single joint_rbm) and new (joint_layers) formats
        if "joint_layers" in payload:
            for rbm in payload["joint_layers"]:
                rbm.to(device)
        elif "joint_rbm" in payload:
            # Backward compatibility: convert single RBM to list
            payload["joint_rbm"].to(device)
            payload["joint_layers"] = [payload["joint_rbm"]]
            payload["num_joint_layers"] = 1

        print(f"[iMDBN_BiModal] Model loaded from {path}")
        if "arch_str" in payload:
            print(f"[iMDBN_BiModal] Architecture: {payload['arch_str']}")

        return payload
