"""
iMDBN (Image-Multimodal Deep Belief Network)

A multimodal extension of iDBN that jointly models images and discrete labels.
Supports cross-modal reconstruction, multimodal training, and unified representation learning.

Key Features:
- Image feature extraction via pretrained iDBN
- Joint RBM connecting image latents and class labels
- Cross-modal reconstruction (image→label, label→image)
- Multimodal training with auxiliary label clamping
- Support for per-class latent statistics (μ-pull)
- Comprehensive evaluation metrics and visualization
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
    compute_joint_embeddings_and_features,
)


class iMDBN(nn.Module):
    """
    Image-Multimodal Deep Belief Network (iMDBN).

    Combines an image-specialized iDBN with a joint RBM that connects:
    - Image latent codes (z_img) from iDBN top layer
    - One-hot encoded class labels (y)

    The joint RBM enables:
    - Cross-modal reconstruction (image↔label)
    - Joint representation learning
    - Multimodal generative modeling

    Supports two constructor signatures:
    1. Long form: iMDBN(image_layers, text_layers, joint_hidden, ...)
    2. Short form: iMDBN(image_layers, joint_hidden, ...)

    Attributes:
        image_idbn: iDBN for image feature extraction
        joint_rbm: RBM connecting image latents and labels
        num_labels: Number of classes
        Dz_img: Dimension of image latent space
        z_class_mean: Per-class mean of image latents (for μ-pull regularization)
        features: Validation set metadata
    """

    def __init__(
        self,
        layer_sizes_img: list,
        layer_sizes_txt_or_joint=None,
        joint_layer_size: Optional[int] = None,
        params: Optional[dict] = None,
        dataloader=None,
        val_loader=None,
        device=None,
        text_posenc_dim: int = 0,
        num_labels: int = 32,
        embedding_dim: int = 64,
        wandb_run=None,
        logging_config_path: Optional[str] = None,
    ):
        """
        Initialize iMDBN.

        Args:
            layer_sizes_img: Image iDBN layer sizes, e.g., [784, 500, 200, 30]
            layer_sizes_txt_or_joint: Either:
                - list/tuple: text layer sizes (legacy API) → joint_layer_size required
                - int: joint hidden size (new API)
            joint_layer_size: Joint RBM hidden size (required if using legacy API)
            params: Configuration dict (same as iDBN)
            dataloader: Training DataLoader yielding (images, one-hot labels)
            val_loader: Validation DataLoader
            device: torch.device
            text_posenc_dim: Ignored (no text RBM in current implementation)
            num_labels: Number of output classes
            embedding_dim: Ignored (kept for API compatibility)
            wandb_run: Optional W&B run for logging
            logging_config_path: Optional path to logging config YAML
        """
        super().__init__()

        # Disambiguate constructor signature
        if isinstance(layer_sizes_txt_or_joint, (list, tuple)):
            # Legacy long form
            if joint_layer_size is None:
                raise ValueError("joint_layer_size required with legacy constructor signature")
        else:
            # New short form
            if joint_layer_size is None:
                joint_layer_size = int(layer_sizes_txt_or_joint)

        self.params = params or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run

        # Load logging config
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

        self.num_labels = int(num_labels)

        # Cache validation batch for snapshots
        try:
            vb_imgs, vb_lbls = next(iter(val_loader))
            self.validation_images = vb_imgs[:8].to(self.device)
            self.validation_labels = vb_lbls[:8].to(self.device)
            self.val_batch = (vb_imgs, vb_lbls)
        except Exception:
            self.validation_images = None
            self.validation_labels = None
            self.val_batch = None

        # Build image iDBN
        self.image_idbn = iDBN(
            layer_sizes=layer_sizes_img,
            params=self.params,
            dataloader=self.dataloader,
            val_loader=self.val_loader,
            device=self.device,
            wandb_run=self.wandb_run,
            logging_config_path=logging_config_path,
        )

        # Build joint RBM
        dz_from_img = int(self.image_idbn.layers[-1].num_hidden)
        self.Dz_img = dz_from_img
        self._build_joint(Dz_img=dz_from_img, joint_hidden=joint_layer_size)

        # Training hyperparameters
        self.joint_cd = int(self.params.get("JOINT_CD", self.params.get("CD", 1)))
        self.cross_steps = int(self.params.get("CROSS_GIBBS_STEPS", 50))
        self.aux_every_k = int(self.params.get("JOINT_AUX_EVERY_K", 0))
        self.aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 50))

        # Extract validation features
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

        self.arch_str = f"IMG{'-'.join(map(str, layer_sizes_img))}_JOINT{joint_layer_size}"

    def _build_joint(self, Dz_img: int, joint_hidden: int):
        """
        Build joint RBM connecting image latents and labels.

        The joint RBM has:
        - Visible layer: [z_img (continuous), y (one-hot as softmax group)]
        - Hidden layer: joint_hidden units

        Args:
            Dz_img: Image latent dimension
            joint_hidden: Joint hidden layer dimension
        """
        self.Dz_img = int(Dz_img)
        K = self.num_labels
        self.joint_rbm = RBM(
            num_visible=self.Dz_img + K,
            num_hidden=int(joint_hidden),
            learning_rate=self.params.get("JOINT_LEARNING_RATE", self.params.get("LEARNING_RATE", 0.1)),
            weight_decay=self.params.get("WEIGHT_PENALTY", 0.0001),
            momentum=self.params.get("INIT_MOMENTUM", 0.5),
            dynamic_lr=self.params.get("LEARNING_RATE_DYNAMIC", True),
            final_momentum=self.params.get("FINAL_MOMENTUM", 0.95),
            softmax_groups=[(self.Dz_img, self.Dz_img + K)],
        ).to(self.device)

    @torch.no_grad()
    def init_joint_bias_from_data(self, n_batches: int = 10):
        """
        Initialize joint RBM visible biases from data statistics.

        Sets:
        - z_img bias: log-odds from empirical p(z_img)
        - y bias: log priors from class frequencies
        - z_class_mean: Per-class mean for later μ-pull regularization

        Args:
            n_batches: Number of batches to scan for statistics
        """
        # Fallback: ensure Dz_img is set
        if not hasattr(self, "Dz_img"):
            if hasattr(self, "joint_rbm"):
                total_v = int(self.joint_rbm.num_visible)
                self.Dz_img = total_v - self.num_labels
            else:
                self.Dz_img = int(self.image_idbn.layers[-1].num_hidden)

        Dz = self.Dz_img
        K = self.num_labels
        sum_z = None
        n = 0
        class_counts = torch.zeros(K, device=self.device)

        # First pass: compute global statistics
        for b, (imgs, lbls) in enumerate(self.dataloader):
            if b >= n_batches:
                break
            v = imgs.to(self.device).view(imgs.size(0), -1).float()
            z = self.image_idbn.represent(v)
            sum_z = z.sum(0) if sum_z is None else (sum_z + z.sum(0))
            n += z.size(0)
            class_counts += lbls.to(self.device).float().sum(0)

        if n == 0:
            return

        mean_z = (sum_z / n).clamp(1e-4, 1 - 1e-4)
        priors = class_counts / max(1, class_counts.sum())
        priors = (priors + 1e-6) / (priors.sum() + 1e-6 * K)

        # Second pass: compute per-class means
        try:
            with torch.no_grad():
                self.z_class_mean = torch.zeros(K, Dz, device=self.device)
                self.z_class_count = torch.zeros(K, device=self.device)

                for b, (imgs, lbls) in enumerate(self.dataloader):
                    if b >= n_batches:
                        break
                    v = imgs.to(self.device).view(imgs.size(0), -1).float()
                    z = self.image_idbn.represent(v)
                    y_idx = lbls.argmax(dim=1)

                    for k in range(K):
                        mask = (y_idx == k)
                        if mask.any():
                            self.z_class_mean[k] += z[mask].sum(0)
                            self.z_class_count[k] += mask.sum()

                # Normalize; fallback to global mean if class not represented
                for k in range(K):
                    if self.z_class_count[k] > 0:
                        self.z_class_mean[k] /= self.z_class_count[k]
                    else:
                        self.z_class_mean[k] = mean_z.clone()

        except Exception as e:
            print(f"[init_joint_bias_from_data] warning: z_class_mean not computed ({e})")
            self.z_class_mean = mean_z.unsqueeze(0).repeat(K, 1)

        # Set joint RBM visible biases
        self.joint_rbm.vis_bias[:Dz] = torch.log(mean_z) - torch.log1p(-mean_z)
        self.joint_rbm.vis_bias[Dz : Dz + K] = torch.log(priors)

    def load_pretrained_image_idbn(self, path: str) -> bool:
        """
        Load pretrained image iDBN from disk.

        Args:
            path: Path to pickled iDBN model

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[load_pretrained_image_idbn] error: {e}")
            return False

        if isinstance(obj, dict) and "layers" in obj:
            self.image_idbn.layers = obj["layers"]
        elif hasattr(obj, "layers"):
            self.image_idbn = obj
            if not hasattr(self.image_idbn, "text_flag"):
                self.image_idbn.text_flag = False
            if not hasattr(self.image_idbn, "arch_dir"):
                self.image_idbn.arch_dir = os.path.join("logs-idbn", "loaded")
                os.makedirs(self.image_idbn.arch_dir, exist_ok=True)
        else:
            print("[load_pretrained_image_idbn] unrecognized format")
            return False

        # Move to device and reinitialize momentum buffers
        for rbm in self.image_idbn.layers:
            rbm.W = rbm.W.to(self.device)
            rbm.hid_bias = rbm.hid_bias.to(self.device)
            rbm.vis_bias = rbm.vis_bias.to(self.device)
            rbm.W_m = torch.zeros_like(rbm.W)
            rbm.hb_m = torch.zeros_like(rbm.hid_bias)
            rbm.vb_m = torch.zeros_like(rbm.vis_bias)
            if not hasattr(rbm, "softmax_groups"):
                rbm.softmax_groups = []

        # Rebuild joint if image latent dimension changed
        dz_pre = int(self.image_idbn.layers[-1].num_hidden)
        if dz_pre != getattr(self, "Dz_img", dz_pre):
            print(f"[load_pretrained_image_idbn] rebuilding joint: Dz_img -> {dz_pre}")
            self._build_joint(Dz_img=dz_pre, joint_hidden=self.joint_rbm.num_hidden)

        print(f"[load_pretrained_image_idbn] loaded from {path}")
        return True

    def finetune_image_last_layer(
        self,
        epochs: int = 0,
        lr_scale: float = 0.3,
        cd_k: Optional[int] = None
    ):
        """
        Fine-tune the last image RBM layer with reduced learning rate.

        Args:
            epochs: Number of fine-tuning epochs (0 = skip)
            lr_scale: Multiplicative scale for learning rate
            cd_k: Contrastive divergence steps (None = use image_idbn.cd_k)
        """
        if epochs <= 0:
            return

        last = self.image_idbn.layers[-1]
        old_lr = float(last.lr)
        last.lr = max(1e-8, old_lr * float(lr_scale))
        use_cd = int(cd_k) if cd_k is not None else int(self.image_idbn.cd_k)

        print(f"[finetune_image_last_layer] epochs={epochs}, lr={last.lr:.4g}, CD={use_cd}")

        for ep in range(int(epochs)):
            losses = []
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.image_idbn.layers[:-1]:
                    v = rbm.forward(v)
                loss = last.train_epoch(v, ep, epochs, CD=use_cd)
                losses.append(float(loss))

            if self.wandb_run and losses:
                self.wandb_run.log({
                    "img_last/finetune_loss": float(np.mean(losses)),
                    "epoch_ft": ep
                })

        last.lr = old_lr
        print("[finetune_image_last_layer] done")

    @torch.no_grad()
    def _cross_reconstruct(
        self,
        z_img: torch.Tensor,
        y_onehot: torch.Tensor,
        steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal reconstruction via joint RBM.

        Performs two sequential operations:
        1. IMG→TXT: Given image latents, infer label distribution
        2. TXT→IMG: Given labels, reconstruct image with per-class μ-pull

        Args:
            z_img: Image latent codes [B, Dz_img]
            y_onehot: One-hot class labels [B, K]
            steps: Number of Gibbs/annealing steps (None = use cross_steps)

        Returns:
            Tuple of:
            - img_reconstructed: Reconstructed images [B, D]
            - p_y_given_img: Predicted label distribution [B, K]
        """
        if steps is None:
            steps = self.cross_steps

        B = z_img.size(0)
        Dz = self.Dz_img
        K = self.num_labels
        V = Dz + K

        # --- IMG→TXT: Image latents → label distribution
        v_known = torch.zeros(B, V, device=self.device)
        km = torch.zeros_like(v_known)
        v_known[:, :Dz] = z_img
        km[:, :Dz] = 1.0

        v_img2txt = self.joint_rbm.conditional_gibbs(
            v_known, km, n_steps=steps, sample_h=False, sample_v=False
        )
        p_y_given_img = v_img2txt[:, Dz:]

        # --- TXT→IMG: Labels → image latents with μ-pull
        v_known.zero_()
        km.zero_()
        v_known[:, Dz:] = y_onehot
        km[:, Dz:] = 1.0

        # Use per-class means for initialization (μ-pull)
        use_mu = hasattr(self, "z_class_mean") and self.z_class_mean is not None
        if use_mu:
            y_idx = y_onehot.argmax(dim=1)
            mu_k = self.z_class_mean[y_idx]
            self.joint_rbm._mu_pull = {"mu_k": mu_k, "eta0": 0.15}
        else:
            self.joint_rbm._mu_pull = None

        # Noisy mean-field annealing with μ-pull
        v_chain = self.joint_rbm.noisy_meanfield_annealed(
            v_known=v_known, known_mask=km,
            n_steps=steps, T0=3.0, T1=1.0, sigma0=0.9, hot_frac=0.7,
            sharpen_last=3, T_cold_plus=0.9
        )

        # Best-of-K refinement (cheap, 5 candidates)
        Kbuf = 5
        buf_v = [v_chain]
        buf_F = [
            self.joint_rbm.free_energy(v_chain)
            if hasattr(self.joint_rbm, "free_energy")
            else torch.zeros(B, device=self.device)
        ]

        for _ in range(Kbuf - 1):
            v_last = self.joint_rbm.noisy_meanfield_annealed(
                v_known=buf_v[-1], known_mask=km,
                n_steps=1, T0=0.9, T1=0.9, sigma0=0.0, hot_frac=0.0,
                sharpen_last=0, T_cold_plus=0.9
            )
            buf_v.append(v_last)
            if hasattr(self.joint_rbm, "free_energy"):
                buf_F.append(self.joint_rbm.free_energy(v_last))
            else:
                buf_F.append(torch.zeros(B, device=self.device))

        buf_F = torch.stack(buf_F, dim=0)
        best_idx = buf_F.argmin(dim=0)
        v_pick = torch.stack([buf_v[int(best_idx[b])][b] for b in range(B)], dim=0)

        self.joint_rbm._mu_pull = None

        z_img_from_y_aff = v_pick[:, :Dz]

        # Optional: apply affine inverse if gain-control was used
        if hasattr(self, "z_affine_scale") and hasattr(self, "z_affine_bias"):
            z_img_from_y = (z_img_from_y_aff - self.z_affine_bias) / (self.z_affine_scale + 1e-6)
        else:
            z_img_from_y = z_img_from_y_aff

        # Decode to image space
        img_from_txt = self.image_idbn.decode(z_img_from_y)
        return img_from_txt, p_y_given_img

    @torch.no_grad()
    def represent(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Compute joint representation from images and labels.

        Args:
            batch: Tuple of (images, one-hot labels)

        Returns:
            Joint hidden layer activations [B, joint_hidden_size]
        """
        img_data, lbl_data = batch
        img = img_data.to(self.device).view(img_data.size(0), -1).float()
        y = lbl_data.to(self.device).float()
        z_img = self.image_idbn.represent(img)
        v = torch.cat([z_img, y], dim=1)
        return self.joint_rbm.forward(v)

    def train_joint(
        self,
        epochs: int,
        log_every_pca: int = 25,
        log_every_probe: int = 10,
        log_every: int = 5,
        w_rec: float = 1.0,
        w_sup: float = 0.0
    ):
        """
        Train joint RBM with multimodal objectives.

        Training strategy:
        - Warmup (first 8 epochs): Label clamping only (strong supervision)
        - Main (remaining epochs): Free CD + auxiliary label/image clamping

        Logged metrics:
        - Cross-modal: text top-1/3 accuracy, text CE, image MSE
        - Embeddings: PCA projections of joint latents
        - Linear probes: Downstream task performance

        Args:
            epochs: Total training epochs
            log_every_pca: Log PCA every N epochs
            log_every_probe: Log linear probes every N epochs
            log_every: Log snapshots every N epochs
            w_rec: Weight for reconstruction loss (unused in current implementation)
            w_sup: Weight for supervision loss (unused in current implementation)
        """
        print("[iMDBN] joint training (with warmup y-clamp)")
        self.init_joint_bias_from_data(n_batches=10)

        WARMUP_Y_EPOCHS = 8

        for epoch in tqdm(range(int(epochs)), desc="Joint"):
            cd_losses = []
            totals = {
                "n": 0,
                "top1": 0,
                "top3": 0,
                "ce_sum": 0.0,
                "mse_sum": 0.0,
                "npix": None
            }

            for b_idx, (img, y) in enumerate(self.dataloader):
                img = img.to(self.device).view(img.size(0), -1).float()
                y = y.to(self.device).float()

                with torch.no_grad():
                    z_img = self.image_idbn.represent(img)
                    v_plus = torch.cat([z_img, y], dim=1)

                B, Dz, K = z_img.size(0), self.Dz_img, self.num_labels
                V = Dz + K

                aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 10))

                if epoch < WARMUP_Y_EPOCHS:
                    # Warmup: label clamping only, 2x per batch
                    for _ in range(2):
                        v_known = torch.zeros(B, V, device=self.device)
                        km = torch.zeros(B, V, device=self.device)
                        v_known[:, Dz:] = y
                        km[:, Dz:] = 1.0
                        _ = self.joint_rbm.train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=1, cond_init_steps=aux_cond_steps,
                            sample_h=False, sample_v=False,
                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )
                else:
                    # Main training: free CD + auxiliary objectives
                    loss_cd = self.joint_rbm.train_epoch(v_plus, epoch, epochs, CD=self.joint_cd)
                    cd_losses.append(float(loss_cd))

                    # Auxiliary label clamping (always, 1x per batch)
                    v_known = torch.zeros(B, V, device=self.device)
                    km = torch.zeros(B, V, device=self.device)
                    v_known[:, Dz:] = y
                    km[:, Dz:] = 1.0
                    _ = self.joint_rbm.train_epoch_clamped(
                        v_known, km, epoch, epochs,
                        CD=1, cond_init_steps=aux_cond_steps,
                        sample_h=False, sample_v=False,
                        reclamp_negative=False,
                        aux_lr_mult=0.3,
                        use_noisy_init=True,
                    )

                    # Auxiliary image clamping every 50 batches
                    if (b_idx % 50) == 0:
                        v_known.zero_()
                        km.zero_()
                        v_known[:, :Dz] = z_img
                        km[:, :Dz] = 1.0
                        _ = self.joint_rbm.train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=1, cond_init_steps=aux_cond_steps,
                            sample_h=False, sample_v=False,
                            reclamp_negative=False,
                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )

                # Cross-modal metrics
                with torch.no_grad():
                    img_from_txt, p_y_given_img = self._cross_reconstruct(
                        z_img, y, steps=self.cross_steps
                    )
                    gt = y.argmax(dim=1)
                    pred = p_y_given_img.argmax(dim=1)
                    topk = min(3, p_y_given_img.size(1))
                    topk_idx = p_y_given_img.topk(k=topk, dim=1).indices

                    ce = F.binary_cross_entropy(
                        p_y_given_img.clamp(1e-6, 1 - 1e-6),
                        F.one_hot(gt, num_classes=p_y_given_img.size(1)).float(),
                        reduction="sum",
                    )
                    npix = img.view(img.size(0), -1).size(1)
                    mse = F.mse_loss(
                        img_from_txt.view_as(img), img, reduction="sum"
                    )

                    totals["n"] += img.size(0)
                    totals["top1"] += (pred == gt).sum().item()
                    totals["top3"] += (topk_idx == gt.unsqueeze(1)).any(dim=1).sum().item()
                    totals["ce_sum"] += float(ce.item())
                    totals["mse_sum"] += float(mse.item())
                    totals["npix"] = npix

            # Log training metrics
            if self.wandb_run and cd_losses:
                self.wandb_run.log({
                    "joint/cd_loss": float(np.mean(cd_losses)),
                    "epoch": epoch
                })

            if self.wandb_run and totals["n"] > 0:
                top1 = totals["top1"] / totals["n"]
                top3 = totals["top3"] / totals["n"]
                ce_mean = totals["ce_sum"] / totals["n"]
                mse_mean = totals["mse_sum"] / max(1, totals["n"] * max(1, totals["npix"] or 1))
                self.wandb_run.log({
                    "cross_modality/text_top1": top1,
                    "cross_modality/text_top3": top3,
                    "cross_modality/text_ce": ce_mean,
                    "cross_modality/image_mse": mse_mean,
                    "epoch": epoch
                })

            # Visual logging and probes
            if self.wandb_run and self.val_loader is not None and self.features is not None:

                # PCA embeddings
                if epoch % log_every_pca == 0:
                    try:
                        E, feats = compute_joint_embeddings_and_features(self)
                        if E.numel() > 0:
                            emb_np = E.numpy()
                            feat_map = {
                                "Cumulative Area": feats["cum_area"].numpy(),
                                "Convex Hull": feats["convex_hull"].numpy(),
                                "Labels": feats["labels"].numpy(),
                            }
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()

                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2, features=feat_map,
                                    arch_name="Joint_top", dist_name="val",
                                    method_name="pca", wandb_run=self.wandb_run,
                                )
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3, features=feat_map,
                                        arch_name="Joint_top", dist_name="val",
                                        method_name="pca", wandb_run=self.wandb_run,
                                    )
                    except Exception as e:
                        self.wandb_run.log({"warn/joint_pca_error": str(e)})

                # Linear probes
                if epoch % log_every_probe == 0:
                    try:
                        log_joint_linear_probe(
                            self,
                            epoch=epoch,
                            n_bins=5, test_size=0.2,
                            steps=1000, lr=1e-2, patience=20, min_delta=0.0,
                            metric_prefix="joint",
                        )
                    except Exception as e:
                        self.wandb_run.log({"warn/joint_probe_error": str(e)})

            # Snapshots
            if epoch % max(1, int(log_every)) == 0:
                self._log_snapshots(epoch)

        print("[iMDBN] joint training finished.")

    @torch.no_grad()
    def _log_snapshots(self, epoch: int, num: int = 8):
        """
        Log reconstruction snapshots and metrics.

        Logs:
        - Cross-reconstruction grids (GT vs reconstructed)
        - Confusion matrix (predicted vs ground truth labels)
        - Image MSE
        - Top-k label prediction probabilities

        Args:
            epoch: Current epoch for logging
            num: Number of validation samples to log
        """
        if self.wandb_run is None or self.validation_images is None or self.validation_labels is None:
            return

        imgs = self.validation_images[:num]
        lbls = self.validation_labels[:num]

        # Cross-modal reconstruction
        zi = self.image_idbn.represent(imgs.view(imgs.size(0), -1))
        img_from_txt, p_y_given_img = self._cross_reconstruct(zi, lbls, steps=self.cross_steps)
        rec = img_from_txt.clamp(0, 1)

        # Reshape to 4D robustly
        if imgs.ndim == 4:
            B, C, H, W = imgs.shape
            imgs4 = imgs
            rec4 = rec.view(B, C, H, W)
        else:
            B = imgs.size(0)
            N = imgs.size(1)
            side = int(round(N ** 0.5))
            if side * side != N:
                C, H, W = 1, N, 1
            else:
                C, H, W = 1, side, side
            imgs4 = imgs.view(B, C, H, W)
            rec4 = rec.view(B, C, H, W)

        # Grid: GT | REC side-by-side
        pair = torch.stack([imgs4.cpu(), rec4.cpu()], dim=1).view(-1, C, H, W)
        grid_pair = vutils.make_grid(pair, nrow=2)
        self.wandb_run.log({
            "snap/image_from_text": wandb.Image(grid_pair.permute(1, 2, 0).numpy()),
            "epoch": epoch
        })

        # Confusion matrix
        class_names = getattr(self, "class_names", None)
        pred = p_y_given_img.argmax(dim=1)
        gt = lbls.argmax(dim=1)

        if class_names and len(class_names) == self.num_labels:
            cm_plot = wandb.plot.confusion_matrix(
                y_true=[class_names[i] for i in gt.cpu().numpy()],
                preds=[class_names[i] for i in pred.cpu().numpy()],
                class_names=class_names,
            )
        else:
            cm_plot = wandb.plot.confusion_matrix(
                y_true=gt.cpu().numpy(),
                preds=pred.cpu().numpy(),
                class_names=[str(i) for i in range(self.num_labels)],
            )
        self.wandb_run.log({"snap/text_confusion": cm_plot, "epoch": epoch})

        # Image MSE
        mse = F.mse_loss(
            imgs4.view(B, -1).to(self.device),
            rec4.view(B, -1).to(self.device)
        ).item()
        self.wandb_run.log({"snap/image_mse": mse, "epoch": epoch})

        # Top-K label probabilities table
        try:
            probs = p_y_given_img.clamp(1e-9, 1).detach().cpu()
            topk = min(2, probs.size(1))
            top_vals, top_inds = probs.topk(topk, dim=1)

            cols = ["idx", "gt_idx", "pred_idx", "p_pred", "p_y_true"]
            if class_names and len(class_names) == self.num_labels:
                cols += ["gt_label", "pred_label"]

            tbl = wandb.Table(columns=cols)
            for i in range(B):
                gt_i = int(gt[i].item())
                pred_i = int(pred[i].item())
                p_pred = float(probs[i, pred_i].item())
                p_gt = float(probs[i, gt_i].item())
                row = [i, gt_i, pred_i, p_pred, p_gt]
                if class_names and len(class_names) == self.num_labels:
                    row += [class_names[gt_i], class_names[pred_i]]
                tbl.add_data(*row)

            self.wandb_run.log({"snap/text_topk": tbl, "epoch": epoch})
        except Exception as e:
            self.wandb_run.log({"warn/snap_topk_table_error": str(e), "epoch": epoch})

    def save_model(self, path: str):
        """
        Save complete iMDBN model to disk.

        Saves in both:
        1. DBN-compatible format (with "layers" key) for auto-detection
        2. Extended iMDBN format with all components for full functionality

        Payload includes:
        - layers: Flattened [image_rbm_1, ..., image_rbm_n, joint_rbm]
        - image_idbn, joint_rbm: Full model objects
        - num_labels, Dz_img, arch_str: Architecture info
        - params, features: Training config and validation metadata
        - Optional: z_class_mean, z_affine_scale, z_affine_bias, class_names
        - metadata: Save timestamp and model type

        Args:
            path: Output file path (.pkl)
        """
        import datetime

        all_layers = list(self.image_idbn.layers) + [self.joint_rbm]

        payload = {
            # DBN-compatible format (auto-detection)
            "layers": all_layers,
            "params": self.params,

            # Extended iMDBN format (full functionality)
            "image_idbn": self.image_idbn,
            "joint_rbm": self.joint_rbm,
            "num_labels": self.num_labels,

            # Architecture info
            "Dz_img": self.Dz_img,
            "arch_str": self.arch_str,

            # Features for analysis
            "features": self.features,

            # Metadata
            "metadata": {
                "saved_at": datetime.datetime.now().isoformat(),
                "model_type": "iMDBN",
                "architecture": self.arch_str,
            }
        }

        # Optional statistics
        if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
            payload["z_class_mean"] = self.z_class_mean

        if hasattr(self, "z_affine_scale") and self.z_affine_scale is not None:
            payload["z_affine_scale"] = self.z_affine_scale

        if hasattr(self, "z_affine_bias") and self.z_affine_bias is not None:
            payload["z_affine_bias"] = self.z_affine_bias

        if hasattr(self, "class_names") and self.class_names is not None:
            payload["class_names"] = self.class_names

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"[iMDBN] Model saved to {path}")
        print(f"[iMDBN] Architecture: {self.arch_str}")
        print(f"[iMDBN] Total layers: {len(all_layers)} (image: {len(self.image_idbn.layers)}, joint: 1)")
        if self.features is not None:
            print(f"[iMDBN] Features saved: {list(self.features.keys())}")

    @staticmethod
    def load_model(path: str, device=None) -> Dict[str, Any]:
        """
        Load iMDBN model from disk.

        Args:
            path: Path to saved .pkl file
            device: Target device (cuda/cpu). If None, auto-detect.

        Returns:
            Dictionary containing:
            - image_idbn: iDBN object
            - joint_rbm: Joint RBM object
            - num_labels: Number of classes
            - params: Training parameters
            - features: Validation features (if saved)
            - Dz_img: Image latent dimension
            - arch_str: Architecture string
            - z_class_mean, z_affine_scale, z_affine_bias: Optional statistics
            - class_names: Optional class names
            - metadata: Save timestamp and model info

        Example:
            >>> data = iMDBN.load_model("model.pkl")
            >>> image_idbn = data["image_idbn"]
            >>> joint_rbm = data["joint_rbm"]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Move model components to device
        if "image_idbn" in payload:
            for rbm in payload["image_idbn"].layers:
                rbm.to(device)

        if "joint_rbm" in payload:
            payload["joint_rbm"].to(device)

        print(f"[iMDBN] Model loaded from {path}")
        if "arch_str" in payload:
            print(f"[iMDBN] Architecture: {payload['arch_str']}")
        if "features" in payload and payload["features"] is not None:
            print(f"[iMDBN] Features loaded: {list(payload['features'].keys())}")
        if "metadata" in payload:
            print(f"[iMDBN] Saved at: {payload['metadata'].get('saved_at', 'unknown')}")

        return payload
