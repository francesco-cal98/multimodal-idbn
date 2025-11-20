import math
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from sklearn.decomposition import PCA

# === utility imports ===
from imdbn.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
from imdbn.utils.probe_utils import (
    log_linear_probe,
    log_joint_linear_probe,
    compute_val_embeddings_and_features,
    compute_joint_embeddings_and_features,
)
from imdbn.utils.energy_utils import run_and_log_fixed_case
from imdbn.utils.conditional_steps import run_and_log_cross_fixed_case, run_and_log_cross_panel, run_and_log_z_mismatch_check
from imdbn.utils.imdbn_logging import (
    log_latent_trajectory_with_recon_panel as _imdbn_log_latent_trajectory_with_recon_panel,
    log_pca3_trajectory as _imdbn_log_pca3_trajectory,
    log_vecdb_neighbors_for_traj as _imdbn_log_vecdb_neighbors_for_traj,
    log_neighbors_images as _imdbn_log_neighbors_images,
    panel_with_gt_and_neighbors as _imdbn_panel_with_gt_and_neighbors,
    panel_gt_vs_decode_neighbors as _imdbn_panel_gt_vs_decode_neighbors,
    log_joint_auto_recon as _imdbn_log_joint_auto_recon,
    ensure_val_bank as _imdbn_ensure_val_bank,
    find_first_val_index_with_label as _imdbn_find_first_val_index_with_label,
    topk_similar_in_latent as _imdbn_topk_similar_in_latent,
)


# -------------------------
# Helpers
# -------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


# -------------------------
# RBM (Bernoulli vis/hidden + gruppi softmax opzionali)
# -------------------------
class RBM(nn.Module):
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        dynamic_lr: bool = False,
        final_momentum: float = 0.97,
        sparsity: bool = False,
        sparsity_factor: float = 0.05,
        softmax_groups: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.dynamic_lr = bool(dynamic_lr)
        self.final_momentum = float(final_momentum)
        self.sparsity = bool(sparsity)
        self.sparsity_factor = float(sparsity_factor)

        # compat vecchi pickle
        self.softmax_groups = softmax_groups or []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(
            torch.randn(self.num_visible, self.num_hidden, device=device) / math.sqrt(max(1, self.num_visible))
        )
        self.hid_bias = nn.Parameter(torch.zeros(self.num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(self.num_visible, device=device))

        # momentum buffers
        self.W_m = torch.zeros_like(self.W)
        self.hb_m = torch.zeros_like(self.hid_bias)
        self.vb_m = torch.zeros_like(self.vis_bias)

    # RBM.forward
    def forward(self, v: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return sigmoid((v @ self.W + self.hid_bias) / max(1e-6, T))

    # RBM._visible_logits
    def _visible_logits(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return (h @ self.W.T + self.vis_bias) / max(1e-6, T)

    # RBM.visible_probs
    def visible_probs(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        logits = self._visible_logits(h, T=T)
        v_prob = torch.sigmoid(logits)
        for s, e in getattr(self, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(logits[:, s:e], dim=1)
        return v_prob


    # ---- sample v ~ p(v|h)
    def sample_visible(self, v_prob: torch.Tensor) -> torch.Tensor:
        v = (v_prob > torch.rand_like(v_prob)).float()
        groups = getattr(self, "softmax_groups", [])
        for s, e in groups:
            probs = v_prob[:, s:e].clamp(1e-8, 1)
            idx = torch.distributions.Categorical(probs=probs).sample()
            v[:, s:e] = 0.0
            v[torch.arange(v.size(0), device=v.device), s + idx] = 1.0
        return v

    # ---- decoder compatibile
    def backward(self, h: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self._visible_logits(h)
        if return_logits:
            return logits
        return self.visible_probs(h)

    @torch.no_grad()
    def backward_sample(self, h: torch.Tensor) -> torch.Tensor:
        return self.sample_visible(self.visible_probs(h))

    # ---- single Gibbs
    @torch.no_grad()
    def gibbs_step(self, v: torch.Tensor, sample_h: bool = True, sample_v: bool = True):
        h_prob = self.forward(v)
        h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
        v_prob = self.visible_probs(h)
        v_next = self.sample_visible(v_prob) if sample_v else v_prob
        return v_next, v_prob, h, h_prob

    # ---- CD-k
    @torch.no_grad()
    def train_epoch(self, data: torch.Tensor, epoch: int, max_epochs: int, CD: int = 1):
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = data.size(0)

        # positive
        pos_h = self.forward(data)
        pos_assoc = data.T @ pos_h

        # negative
        h = (pos_h > torch.rand_like(pos_h)).float()
        for _ in range(int(CD)):
            v_prob = self.visible_probs(h)
            v = self.sample_visible(v_prob)
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float()
        neg_assoc = v.T @ h_prob

        # updates
        self.W_m.mul_(mom).add_(lr * ((pos_assoc - neg_assoc) / bsz - self.weight_decay * self.W))
        self.W.add_(self.W_m)

        self.hb_m.mul_(mom).add_(lr * (pos_h.sum(0) - h_prob.sum(0)) / bsz)
        if self.sparsity:
            Q = pos_h.mean(0)
            self.hb_m.add_(-lr * (Q - self.sparsity_factor))
        self.hid_bias.add_(self.hb_m)

        self.vb_m.mul_(mom).add_(lr * (data.sum(0) - v.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        loss = torch.mean((data - v_prob) ** 2)
        return loss

    # ---- Gibbs condizionale (re-clamp ad ogni step)
    @torch.no_grad()
    def conditional_gibbs_annealed(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        n_steps: int = 40,
        T0: float = 2.5,
        T1: float = 1.0,
        sample_h_until: int = 20,
        sample_v_every: int = 0,
        final_meanfield: bool = True,
    ):
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)

        # armonizza sample_h_until con n_steps (evita mismatch silenziosi)
        hot_steps = int(max(0, min(n_steps, sample_h_until)))

        for t in range(int(n_steps)):
            Tt = self._lin_schedule(t, n_steps, T0, T1)
            # leggero sharpen negli ultimi step
            if (n_steps - t) <= 3:
                Tt = min(0.9, Tt)

            h_prob = self.forward(v, T=Tt)
            h = (h_prob > torch.rand_like(h_prob)).float() if t < hot_steps else h_prob

            v_prob = self.visible_probs(h, T=Tt)
            if (t < hot_steps) and (sample_v_every > 0) and (t % sample_v_every == 0):
                v_new = self.sample_visible(v_prob)
            else:
                v_new = v_prob

            v = v_new * (1 - km) + v_known * km

        if final_meanfield:
            h_prob = self.forward(v, T=1.0)
            v = self.visible_probs(h_prob, T=1.0) * (1 - km) + v_known * km

        return v

    def _lin_schedule(self, t, t_max, start, end):
        if t_max <= 1:
            return float(end)
        alpha = min(max(t / (t_max - 1), 0.0), 1.0)
        return float(start + (end - start) * alpha)

    def _hot_steps(self, n_steps, hot_frac):
        return int(max(0, min(n_steps, round(hot_frac * n_steps))))

    @torch.no_grad()
    def noisy_meanfield_annealed(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        n_steps: int = 72,
        T0: float = 3.0,
        T1: float = 1.0,
        sigma0: float = 0.9,
        hot_frac: float = 0.7,
        sharpen_last: int = 3,
        T_cold_plus: float = 0.9,
    ):
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)

        hot_steps = self._hot_steps(n_steps, hot_frac)

        for t in range(int(n_steps)):
            Tt = self._lin_schedule(t, n_steps, T0, T1)
            if (n_steps - t) <= max(1, int(sharpen_last)):
                Tt = T_cold_plus
            sig_t = sigma0 * max(0.0, 1.0 - (t / max(1, n_steps - 1)))

            # h|v con rumore sui LOGIT
            h_logits = (v @ self.W + self.hid_bias) / max(1e-6, Tt)
            if sig_t > 0:
                h_logits = h_logits + torch.randn_like(h_logits) * sig_t
            h_prob = torch.sigmoid(h_logits)

            # v|h con rumore sui LOGIT
            v_logits = (h_prob @ self.W.T + self.vis_bias) / max(1e-6, Tt)
            if sig_t > 0:
                v_logits = v_logits + torch.randn_like(v_logits) * sig_t

            v_prob = torch.sigmoid(v_logits)
            for s, e in getattr(self, "softmax_groups", []):
                v_prob[:, s:e] = torch.softmax(v_logits[:, s:e], dim=1)

            # opzionale: μ-pull (se impostato dal chiamante)
            if hasattr(self, "_mu_pull") and self._mu_pull is not None:
                Dz = self._mu_pull["mu_k"].size(1)
                eta0 = float(self._mu_pull.get("eta0", 0.15))
                eta_t = eta0 * max(0.0, 1.0 - (t / max(1, n_steps - 1)))
                # spingi i logit (via differenza prob) leggermente verso μ_k
                v_prob[:, :Dz] = (1 - eta_t) * v_prob[:, :Dz] + eta_t * self._mu_pull["mu_k"]

            v = v_prob * (1 - km) + v_known * km

        return v

    @torch.no_grad()
    def train_epoch_clamped(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        epoch: int,
        max_epochs: int,
        CD: int = 1,
        cond_init_steps: int = 50,
        sample_h: bool = True,
        sample_v: bool = False,
        # --- nuovi parametri ---
        reclamp_negative: bool = True,     # se False la fase negativa è "libera"
        aux_lr_mult: float = 0.3,          # scala l'intensità dell'update aux
        use_noisy_init: bool = True,       # init positivo più robusto
    ):
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = v_known.size(0)

        if use_noisy_init:
            v_plus = self.noisy_meanfield_annealed(
                v_known=v_known, known_mask=known_mask,
                n_steps=max(10, int(cond_init_steps)),
                T0=3.0, T1=1.0, sigma0=0.9, hot_frac=0.7, sharpen_last=2, T_cold_plus=0.9
            )
        else:
            v_plus = self.conditional_gibbs(
                v_known, known_mask, n_steps=cond_init_steps,
                sample_h=sample_h, sample_v=sample_v
            )

        h_plus = self.forward(v_plus)
        pos_assoc = v_plus.T @ h_plus

        v_neg = v_plus.clone()
        for _ in range(int(CD)):
            h_prob = self.forward(v_neg)
            h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
            v_prob = self.visible_probs(h)
            if reclamp_negative:
                v_neg = v_prob * (1 - known_mask) + v_known * known_mask
            else:
                v_neg = v_prob
            if sample_v:
                v_neg = self.sample_visible(v_neg)

        h_neg = self.forward(v_neg)
        neg_assoc = v_neg.T @ h_neg

        wd_term = self.weight_decay * self.W
        self.W_m.mul_(mom).add_(aux_lr_mult * lr * ((pos_assoc - neg_assoc) / bsz - wd_term))
        self.W.add_(self.W_m)
        self.hb_m.mul_(mom).add_(aux_lr_mult * lr * (h_plus.sum(0) - h_neg.sum(0)) / bsz)
        self.hid_bias.add_(self.hb_m)
        self.vb_m.mul_(mom).add_(aux_lr_mult * lr * (v_plus.sum(0) - v_neg.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        return torch.mean((v_plus - v_neg) ** 2)



    @torch.no_grad()
    def conditional_gibbs(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        n_steps: int = 30,
        sample_h: bool = False,
        sample_v: bool = False,
    ) -> torch.Tensor:
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)
        for _ in range(int(n_steps)):
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
            v_prob = self.visible_probs(h)
            v = v_prob * (1 - km) + v_known * km
            if sample_v:
                v = self.sample_visible(v) * (1 - km) + v_known * km
        return self.visible_probs(self.forward(v))

    # ---- clamped-CD (aux)

# -------------------------
# iDBN (immagini) — con PCA/Probes/AutoRecon
# -------------------------
class iDBN:
    def __init__(self, layer_sizes, params, dataloader, val_loader, device, wandb_run=None, logging_config_path: Optional[str] = None):
        self.layers: List[RBM] = []
        self.params = params
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run
        # load logging config if present
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

        # campi attesi dalle tue utils
        self.text_flag = False
        self.arch_str = "-".join(map(str, layer_sizes))
        self.arch_dir = os.path.join("logs-idbn", f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self.cd_k = int(self.params.get("CD", 1))
        self.sparsity_last = bool(self.params.get("SPARSITY", False))
        self.sparsity_factor = float(self.params.get("SPARSITY_FACTOR", 0.1))

        # cache val
        try:
            self.val_batch, self.val_labels = next(iter(val_loader))
        except Exception:
            self.val_batch, self.val_labels = None, None

        # features complete (no shuffle sul val_loader!)
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

        # costruzione RBM
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

    # quali layer monitorare (come nei tuoi log)
    def _layers_to_monitor(self) -> List[int]:
        layers = {len(self.layers)}
        if len(self.layers) > 1:
            layers.add(1)
        return sorted(layers)

    def _layer_tag(self, idx: int) -> str:
        return f"layer{idx}"

    # TRAIN con autorecon + PCA + probes
    def train(self, epochs: int, log_every_pca: int = 25, log_every_probe: int = 10):
        for epoch in tqdm(range(int(epochs)), desc="iDBN"):
            losses = []
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.layers:
                    loss = rbm.train_epoch(v, epoch, epochs, CD=self.cd_k)
                    v = rbm.forward(v)
                    losses.append(float(loss))
            if self.wandb_run and losses:
                self.wandb_run.log({"idbn/loss": float(np.mean(losses)), "epoch": epoch})

            # Auto-recon snapshot
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
                self.wandb_run.log({"idbn/auto_recon_grid": wandb.Image(grid.permute(1, 2, 0).numpy()), "epoch": epoch})
                try:
                    mse = F.mse_loss(img0.to(self.device).view(img0.size(0), -1), recv.view(img0.size(0), -1))
                    self.wandb_run.log({"idbn/auto_recon_mse": mse.item(), "epoch": epoch})
                except Exception:
                    pass

            # PCA + PROBES per-layer
            if self.wandb_run and self.val_loader is not None and self.features is not None:
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
        v = x.view(x.size(0), -1).float().to(self.device)
        L = len(self.layers) if (upto_layer is None) else max(0, min(len(self.layers), int(upto_layer)))
        for i in range(L):
            v = self.layers[i].forward(v)
        return v

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        cur = v
        for rbm in self.layers:
            cur = rbm.forward(cur)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def decode(self, top: torch.Tensor) -> torch.Tensor:
        cur = top.to(self.device)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def save_model(self, path: str):
        model_copy = {"layers": self.layers, "params": self.params}
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)


# -------------------------
# iMDBN (multimodale) — semplice, etichette come softmax block
# -------------------------
class iMDBN(nn.Module):
    """
    Joint RBM su [z_img  ⊕  y_onehot] con y gestito come gruppo softmax.
    Supporta:
      - forma lunga: iMDBN(image_layers, text_layers, joint_hidden, params, ...)
      - forma corta: iMDBN(image_layers, joint_hidden, params, ...)
    """
    def __init__(
        self,
        layer_sizes_img,
        layer_sizes_txt_or_joint=None,          # può essere list/tuple (vecchia API) o int (nuova API)
        joint_layer_size=None,
        params=None,
        dataloader=None,
        val_loader=None,
        device=None,
        text_posenc_dim: int = 0,               # ignorato (no TextRBM)
        num_labels: int = 32,
        embedding_dim: int = 64,
        wandb_run=None,
        logging_config_path: Optional[str] = None,
    ):
        super().__init__()

        # disambiguazione firma
        if isinstance(layer_sizes_txt_or_joint, (list, tuple)):
            # forma lunga
            layer_sizes_txt_unused = layer_sizes_txt_or_joint
            if joint_layer_size is None:
                raise ValueError("joint_layer_size mancante nella forma lunga.")
        else:
            # forma corta
            if joint_layer_size is None:
                joint_layer_size = int(layer_sizes_txt_or_joint)
            layer_sizes_txt_unused = None

        self.params = params or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        # logging config (optional)
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

        # snapshot val
        try:
            vb_imgs, vb_lbls = next(iter(val_loader))
            self.validation_images = vb_imgs[:8].to(self.device)
            self.validation_labels = vb_lbls[:8].to(self.device)
            self.val_batch = (vb_imgs, vb_lbls)
        except Exception:
            self.validation_images = None
            self.validation_labels = None
            self.val_batch = None

        # iDBN immagine
        self.image_idbn = iDBN(
            layer_sizes=layer_sizes_img,
            params=self.params,
            dataloader=self.dataloader,
            val_loader=self.val_loader,
            device=self.device,
            wandb_run=self.wandb_run,
            logging_config_path=logging_config_path,
        )

        # dimensione top immagine + joint
        dz_from_img = int(self.image_idbn.layers[-1].num_hidden)
        self.Dz_img = dz_from_img
        self._build_joint(Dz_img=dz_from_img, joint_hidden=joint_layer_size)

        # knobs
        self.joint_cd = int(self.params.get("JOINT_CD", self.params.get("CD", 1)))
        self.cross_steps = int(self.params.get("CROSS_GIBBS_STEPS", 50))
        self.aux_every_k = int(self.params.get("JOINT_AUX_EVERY_K", 0))
        self.aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 50))

        # features joint complete (no shuffle!)
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

        self.arch_str = f"IMG{'-'.join(map(str,layer_sizes_img))}_JOINT{joint_layer_size}"

    def _build_joint(self, Dz_img: int, joint_hidden: int):
        self.Dz_img = int(Dz_img)  # sempre presente
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

    
    # ---- init bias visibili del joint
    @torch.no_grad()
    def init_joint_bias_from_data(self, n_batches: int = 10):
        # fallback robusto
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
        try:
            with torch.no_grad():
                self.z_class_mean = torch.zeros(K, Dz, device=self.device)
                self.z_class_count = torch.zeros(K, device=self.device)
                # seconda passata leggera: accumula per classe (limita a n_batches per costo)
                processed = 0
                for b, (imgs, lbls) in enumerate(self.dataloader):
                    if b >= n_batches:
                        break
                    v = imgs.to(self.device).view(imgs.size(0), -1).float()
                    z = self.image_idbn.represent(v)  # [B, Dz]
                    y_idx = lbls.argmax(dim=1)        # [B]
                    for k in range(K):
                        mask = (y_idx == k)
                        if mask.any():
                            self.z_class_mean[k] += z[mask].sum(0)
                            self.z_class_count[k] += mask.sum()
                    processed += z.size(0)
                # normalizza; fallback alla media globale se una classe ha 0 conteggi
                for k in range(K):
                    if self.z_class_count[k] > 0:
                        self.z_class_mean[k] /= self.z_class_count[k]
                    else:
                        self.z_class_mean[k] = mean_z.clone()
        except Exception as e:
            print(f"[init_joint_bias_from_data] warn: z_class_mean not computed ({e})")
            # fallback: media unica per tutte le classi
            self.z_class_mean = mean_z.unsqueeze(0).repeat(K, 1)

        
        self.joint_rbm.vis_bias[:Dz] = torch.log(mean_z) - torch.log1p(-mean_z)
        self.joint_rbm.vis_bias[Dz : Dz + K] = torch.log(priors)

    # ---- load iDBN pre-allenata
    def load_pretrained_image_idbn(self, path: str) -> bool:
        import pickle
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[load_pretrained_image_idbn] errore: {e}")
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
            print("[load_pretrained_image_idbn] formato non riconosciuto")
            return False

        # porta su device e init buffers
        for rbm in self.image_idbn.layers:
            rbm.W = rbm.W.to(self.device)
            rbm.hid_bias = rbm.hid_bias.to(self.device)
            rbm.vis_bias = rbm.vis_bias.to(self.device)
            rbm.W_m = torch.zeros_like(rbm.W)
            rbm.hb_m = torch.zeros_like(rbm.hid_bias)
            rbm.vb_m = torch.zeros_like(rbm.vis_bias)
            if not hasattr(rbm, "softmax_groups"):
                rbm.softmax_groups = []

        dz_pre = int(self.image_idbn.layers[-1].num_hidden)
        if dz_pre != getattr(self, "Dz_img", dz_pre):
            print(f"[load_pretrained_image_idbn] adatto joint: Dz_img -> {dz_pre}")
            self._build_joint(Dz_img=dz_pre, joint_hidden=self.joint_rbm.num_hidden)

        print(f"[load_pretrained_image_idbn] caricato {path}")
        return True

    # ---- fine tuning leggero dell'ultimo RBM immagine (opzionale)
    def finetune_image_last_layer(self, epochs: int = 0, lr_scale: float = 0.3, cd_k: Optional[int] = None):
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
                self.wandb_run.log({"img_last/finetune_loss": float(np.mean(losses)), "epoch_ft": ep})
        last.lr = old_lr
        print("[finetune_image_last_layer] done")

    # ---- cross reconstruction


    @torch.no_grad()
    def _cross_reconstruct(self, z_img: torch.Tensor, y_onehot: torch.Tensor, steps: Optional[int] = None):
        if steps is None:
            steps = self.cross_steps
        B = z_img.size(0)
        Dz = self.Dz_img
        K = self.num_labels
        V = Dz + K

        # IMG -> TXT (come prima)
        v_known = torch.zeros(B, V, device=self.device)
        km = torch.zeros_like(v_known)
        v_known[:, :Dz] = z_img
        km[:, :Dz] = 1.0
        v_img2txt = self.joint_rbm.conditional_gibbs(v_known, km, n_steps=steps, sample_h=False, sample_v=False)
        p_y_given_img = v_img2txt[:, Dz:]

        # TXT -> IMG (noisy MF + μ-pull + best-of-K)
        v_known.zero_(); km.zero_()
        v_known[:, Dz:] = y_onehot
        km[:, Dz:] = 1.0

        # init informata con μ_k, se disponibile
        use_mu = hasattr(self, "z_class_mean") and self.z_class_mean is not None
        if use_mu:
            y_idx = y_onehot.argmax(dim=1)
            mu_k = self.z_class_mean[y_idx]  # [B, Dz]
            # abilita μ-pull nel RBM durante l'annealing
            self.joint_rbm._mu_pull = {"mu_k": mu_k, "eta0": 0.15}
        else:
            self.joint_rbm._mu_pull = None

        # catena principale
        v_chain = self.joint_rbm.noisy_meanfield_annealed(
            v_known=v_known, known_mask=km,
            n_steps=steps, T0=3.0, T1=1.0, sigma0=0.9, hot_frac=0.7,
            sharpen_last=3, T_cold_plus=0.9
        )

        # best-of-K “freddo” (cheap): 1-step refinement ripetuto Kbuf volte
        Kbuf = 5
        buf_v = [v_chain]
        buf_F = [self.joint_rbm.free_energy(v_chain) if hasattr(self.joint_rbm, "free_energy") else torch.zeros(B, device=self.device)]
        for _ in range(Kbuf - 1):
            v_last = self.joint_rbm.noisy_meanfield_annealed(
                v_known=buf_v[-1], known_mask=km,
                n_steps=1, T0=0.9, T1=0.9, sigma0=0.0, hot_frac=0.0, sharpen_last=0, T_cold_plus=0.9
            )
            buf_v.append(v_last)
            if hasattr(self.joint_rbm, "free_energy"):
                buf_F.append(self.joint_rbm.free_energy(v_last))
            else:
                buf_F.append(torch.zeros(B, device=self.device))

        buf_F = torch.stack(buf_F, dim=0)  # [K,B]
        best_idx = buf_F.argmin(dim=0)
        v_pick = torch.stack([buf_v[int(best_idx[b])][b] for b in range(B)], dim=0)

        self.joint_rbm._mu_pull = None  # cleanup

        z_img_from_y_aff = v_pick[:, :Dz]

        # se usi gain-control affine su z, **inversa** qui (altrimenti lascia così)
        if hasattr(self, "z_affine_scale") and hasattr(self, "z_affine_bias"):
            z_img_from_y = (z_img_from_y_aff - self.z_affine_bias) / (self.z_affine_scale + 1e-6)
        else:
            z_img_from_y = z_img_from_y_aff

        img_from_txt = self.image_idbn.decode(z_img_from_y)
        return img_from_txt, p_y_given_img


    # ---- rappresentazione top (joint hidden)
    @torch.no_grad()
    def represent(self, batch):
        img_data, lbl_data = batch
        img = img_data.to(self.device).view(img_data.size(0), -1).float()
        y = lbl_data.to(self.device).float()
        z_img = self.image_idbn.represent(img)
        v = torch.cat([z_img, y], dim=1)
        return self.joint_rbm.forward(v)

    def train_joint(self, epochs, log_every_pca: int = 25, log_every_probe: int = 10, log_every: int = 5, w_rec: float = 1.0, w_sup: float = 0.0):
        print("[iMDBN] joint training (with warmup y-clamp)")
        self.init_joint_bias_from_data(n_batches=10)

        WARMUP_Y_EPOCHS = 8
        for epoch in tqdm(range(int(epochs)), desc="Joint"):
            cd_losses = []
            totals = {"n": 0, "top1": 0, "top3": 0, "ce_sum": 0.0, "mse_sum": 0.0, "npix": None}

            for b_idx, (img, y) in enumerate(self.dataloader):
                img = img.to(self.device).view(img.size(0), -1).float()
                y = y.to(self.device).float()

                with torch.no_grad():
                    z_img = self.image_idbn.represent(img)
                    v_plus = torch.cat([z_img, y], dim=1)

                B, Dz, K = z_img.size(0), self.Dz_img, self.num_labels
                V = Dz + K

                # Usa parametro dal config invece di hardcoded
                aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 10))

                if epoch < WARMUP_Y_EPOCHS:
                    # --- Warmup: SOLO y-clamp, 2x per batch, nessun CD libero
                    for _ in range(2):
                        v_known = torch.zeros(B, V, device=self.device); km = torch.zeros(B, V, device=self.device)
                        v_known[:, Dz:] = y; km[:, Dz:] = 1.0
                        _ = self.joint_rbm.train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=1, cond_init_steps=aux_cond_steps,  # Ora usa config (10 invece di 50)
                            sample_h=False, sample_v=False,

                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )
                else:
                    # --- CD "freddo"
                    loss_cd = self.joint_rbm.train_epoch(v_plus, epoch, epochs, CD=self.joint_cd)
                    cd_losses.append(float(loss_cd))

                    # --- Aux y-clamp SEMPRE (1x per batch)
                    v_known = torch.zeros(B, V, device=self.device); km = torch.zeros(B, V, device=self.device)
                    v_known[:, Dz:] = y; km[:, Dz:] = 1.0
                    _ = self.joint_rbm.train_epoch_clamped(
                        v_known, km, epoch, epochs,
                        CD=1, cond_init_steps=aux_cond_steps,  # Ora usa config (10 invece di 50)
                        sample_h=False, sample_v=False,
                        reclamp_negative=False,
                        aux_lr_mult=0.3,
                        use_noisy_init=True,
                    )

                    # --- (opzionale) Aux image-clamp ogni ~50 batch (ridotto da 12)
                    if (b_idx % 50) == 0:
                        v_known.zero_(); km.zero_()
                        v_known[:, :Dz] = z_img; km[:, :Dz] = 1.0
                        _ = self.joint_rbm.train_epoch_clamped(
                            v_known, km, epoch, epochs,
                            CD=1, cond_init_steps=aux_cond_steps,  # Ora usa config (10 invece di 50)
                            sample_h=False, sample_v=False,
                            reclamp_negative=False,
                            aux_lr_mult=0.3,
                            use_noisy_init=True,
                        )

                # --- Cross-modality metrics online (immutato)
                with torch.no_grad():
                    img_from_txt, p_y_given_img = self._cross_reconstruct(z_img, y, steps=self.cross_steps)
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
                    mse = F.mse_loss(img_from_txt.view_as(img), img, reduction="sum")

                    totals["n"] += img.size(0)
                    totals["top1"] += (pred == gt).sum().item()
                    totals["top3"] += (topk_idx == gt.unsqueeze(1)).any(dim=1).sum().item()
                    totals["ce_sum"] += float(ce.item())
                    totals["mse_sum"] += float(mse.item())
                    totals["npix"] = npix

            if self.wandb_run and cd_losses:
                self.wandb_run.log({"joint/cd_loss": float(np.mean(cd_losses)), "epoch": epoch})
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

            # logging visuale/sonde come già avevi
            if self.wandb_run and self.val_loader is not None and self.features is not None:
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

                if epoch % 5 == 0:
                    run_and_log_cross_fixed_case(self, epoch=epoch, target_label=29,
                                                max_steps=self.cross_steps, sample_h=False, sample_v=False,
                                                tag="fixed_lbl12")
                    run_and_log_z_mismatch_check(self, epoch=epoch, max_steps=self.cross_steps, sample_h=False, sample_v=False, tag="val")
                    sample_idx = _imdbn_find_first_val_index_with_label(self, 2)
                    _imdbn_log_vecdb_neighbors_for_traj(self, sample_idx=sample_idx, steps=self.cross_steps,
                                                        k=8, metric="cosine", tag="vecdb",
                                                        also_l2=True, dedup="image", exclude_self=True)

                if epoch % 10 == 0:
                    run_and_log_cross_panel(self, epoch=epoch, per_class=4, max_steps=self.cross_steps,
                                            sample_h=False, sample_v=False, tag="panel_4_per_class")
                    idx4 = _imdbn_find_first_val_index_with_label(self, 4)
                    if idx4 >= 0:
                        from imdbn.utils.imdbn_logging import log_pca3_trajectory_with_recon_panel as _imdbn_log_pca3_traj_panel
                        _imdbn_log_pca3_traj_panel(self, sample_idx=idx4, steps=self.cross_steps, tag="pca3_lbl4")
                    _imdbn_log_latent_trajectory_with_recon_panel(self, sample_idx=idx4, steps=self.cross_steps, tag="sanity_pca_traj")

            if epoch % max(1, int(log_every)) == 0:
                self._log_snapshots(epoch)
                _imdbn_log_joint_auto_recon(self, epoch)

        print("[iMDBN] joint training finished.")


    @torch.no_grad()
    def _log_snapshots(self, epoch, num=8):
        if self.wandb_run is None or self.validation_images is None or self.validation_labels is None:
            return

        imgs = self.validation_images[:num]
        lbls = self.validation_labels[:num]

        # --- IMG->TXT->IMG cross reconstruction ---
        zi = self.image_idbn.represent(imgs.view(imgs.size(0), -1))
        img_from_txt, p_y_given_img = self._cross_reconstruct(zi, lbls, steps=self.cross_steps)
        rec = img_from_txt.clamp(0, 1)  # [B, N] o [B, C*H*W]

        # --- rendi entrambe le immagini 4D (B, C, H, W) in modo robusto
        if imgs.ndim == 4:
            B, C, H, W = imgs.shape
            imgs4 = imgs
            rec4  = rec.view(B, C, H, W)
        else:
            B = imgs.size(0)
            N = imgs.size(1)
            side = int(round(N ** 0.5))
            if side * side != N:
                # fallback: evento raro → tratta come (C=1, H=N, W=1)
                C, H, W = 1, N, 1
            else:
                C, H, W = 1, side, side
            imgs4 = imgs.view(B, C, H, W)
            rec4  = rec.view(B, C, H, W)

        # === GT | REC in un'unica griglia (2 colonne) ===
        pair = torch.stack([imgs4.cpu(), rec4.cpu()], dim=1).view(-1, C, H, W)
        grid_pair = vutils.make_grid(pair, nrow=2)
        self.wandb_run.log({
            "snap/image_from_text": wandb.Image(grid_pair.permute(1, 2, 0).numpy()),
            "epoch": epoch
        })

        # --- Confusion matrix (usa class_names se disponibile)
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

        # --- MSE immagine (flatten coerente)
        mse = F.mse_loss(imgs4.view(B, -1).to(self.device), rec4.view(B, -1).to(self.device)).item()
        self.wandb_run.log({"snap/image_mse": mse, "epoch": epoch})

        # === Tabella TOP-K per le probabilità testo ===
        # (per ogni sample nello snapshot: gt/pred con indici e nomi, p(pred), p(y_true))
        try:
            probs = p_y_given_img.clamp(1e-9, 1).detach().cpu()  # [B,K]
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
                p_gt   = float(probs[i, gt_i].item())
                row = [i, gt_i, pred_i, p_pred, p_gt]
                if class_names and len(class_names) == self.num_labels:
                    row += [class_names[gt_i], class_names[pred_i]]
                tbl.add_data(*row)
            self.wandb_run.log({ "snap/text_topk": tbl, "epoch": epoch })
        except Exception as e:
            self.wandb_run.log({ "warn/snap_topk_table_error": str(e), "epoch": epoch })
    
    

    def save_model(self, path: str):
        """
        Save complete iMDBN model compatible with numerical_analysis_pipeline.

        Saves in TWO formats:
        1. DBN-compatible format with "layers" attribute (for auto-detection)
        2. Extended iMDBN format with all components (for full functionality)

        The "layers" list contains: [image_rbm_1, image_rbm_2, ..., joint_rbm]
        This allows DBNAdapter to load it automatically.

        Saves:
        - layers: Flattened list of all RBMs (image_idbn.layers + joint_rbm)
        - params: Training hyperparameters
        - image_idbn: Complete image DBN with all layers
        - joint_rbm: Joint RBM layer
        - num_labels: Number of output classes
        - features: Validation features (Cumulative Area, Convex Hull, Density, Labels)
        - Dz_img: Dimension of image latent space
        - arch_str: Architecture string (e.g., "IMG10000-1500-500_JOINT256")
        - Optional statistics: z_class_mean, z_affine_scale, z_affine_bias
        - Optional: class_names if available
        - Metadata: timestamp, training info
        """
        import datetime

        # Create flattened layers list for DBN compatibility
        # This allows numerical_analysis_pipeline's DBNAdapter to auto-detect the model
        all_layers = list(self.image_idbn.layers) + [self.joint_rbm]

        payload = {
            # DBN-compatible format (for auto-detection)
            "layers": all_layers,
            "params": self.params,

            # Extended iMDBN format (for full functionality)
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

        # Optional statistics (for cross-reconstruction)
        if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
            payload["z_class_mean"] = self.z_class_mean

        if hasattr(self, "z_affine_scale") and self.z_affine_scale is not None:
            payload["z_affine_scale"] = self.z_affine_scale

        if hasattr(self, "z_affine_bias") and self.z_affine_bias is not None:
            payload["z_affine_bias"] = self.z_affine_bias

        # Optional class names
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
    def load_model(path: str, device=None):
        """
        Load iMDBN model from disk.

        Args:
            path: Path to saved .pkl file
            device: Target device (cuda/cpu). If None, auto-detect from model.

        Returns:
            Dictionary with all saved components:
            - image_idbn: Image DBN
            - joint_rbm: Joint RBM
            - num_labels: Number of classes
            - params: Training parameters
            - features: Validation features (if saved)
            - Dz_img: Image latent dimension
            - arch_str: Architecture string
            - z_class_mean, z_affine_scale, z_affine_bias: Optional statistics
            - class_names: Optional class names
            - metadata: Save timestamp and info

        Example:
            >>> model_data = iMDBN.load_model("path/to/model.pkl")
            >>> image_idbn = model_data["image_idbn"]
            >>> joint_rbm = model_data["joint_rbm"]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Move to device if needed
        if "image_idbn" in payload:
            payload["image_idbn"].to(device)
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
