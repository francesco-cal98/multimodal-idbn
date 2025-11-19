"""
Restricted Boltzmann Machine (RBM) with optional softmax groups.

Supports:
- Bernoulli visible and hidden units
- Optional softmax groups for categorical variables (e.g., labels)
- Contrastive Divergence (CD) training
- Conditional Gibbs sampling with annealing
- Noisy mean-field inference for robust reconstruction
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid function."""
    return 1 / (1 + torch.exp(-x))


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine with Bernoulli units and optional softmax groups.

    Args:
        num_visible: Number of visible units
        num_hidden: Number of hidden units
        learning_rate: Base learning rate for CD updates
        weight_decay: L2 weight decay coefficient
        momentum: Initial momentum for parameter updates
        dynamic_lr: If True, decrease learning rate over epochs
        final_momentum: Momentum value after warmup (default: 0.97)
        sparsity: Enable sparsity regularization on hidden units
        sparsity_factor: Target average activation for sparsity
        softmax_groups: List of (start, end) indices for softmax groups in visible layer
    """

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

        # Softmax groups for categorical variables (e.g., one-hot labels)
        self.softmax_groups = softmax_groups or []

        # Initialize parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(
            torch.randn(self.num_visible, self.num_hidden, device=device) / math.sqrt(max(1, self.num_visible))
        )
        self.hid_bias = nn.Parameter(torch.zeros(self.num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(self.num_visible, device=device))

        # Momentum buffers
        self.W_m = torch.zeros_like(self.W)
        self.hb_m = torch.zeros_like(self.hid_bias)
        self.vb_m = torch.zeros_like(self.vis_bias)

    def forward(self, v: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        Compute p(h|v) with optional temperature.

        Args:
            v: Visible units [batch_size, num_visible]
            T: Temperature for sampling (default: 1.0)

        Returns:
            Hidden probabilities [batch_size, num_hidden]
        """
        return sigmoid((v @ self.W + self.hid_bias) / max(1e-6, T))

    def _visible_logits(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """Compute visible logits given hidden units."""
        return (h @ self.W.T + self.vis_bias) / max(1e-6, T)

    def visible_probs(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        Compute p(v|h) with softmax groups.

        Args:
            h: Hidden units [batch_size, num_hidden]
            T: Temperature for sampling

        Returns:
            Visible probabilities [batch_size, num_visible]
        """
        logits = self._visible_logits(h, T=T)
        v_prob = torch.sigmoid(logits)

        # Apply softmax to categorical groups
        for s, e in getattr(self, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(logits[:, s:e], dim=1)

        return v_prob

    def sample_visible(self, v_prob: torch.Tensor) -> torch.Tensor:
        """
        Sample v ~ p(v|h) respecting softmax groups.

        For Bernoulli units: sample from Bernoulli(v_prob)
        For softmax groups: sample from Categorical distribution
        """
        v = (v_prob > torch.rand_like(v_prob)).float()

        # Sample from categorical distribution for softmax groups
        groups = getattr(self, "softmax_groups", [])
        for s, e in groups:
            probs = v_prob[:, s:e].clamp(1e-8, 1)
            idx = torch.distributions.Categorical(probs=probs).sample()
            v[:, s:e] = 0.0
            v[torch.arange(v.size(0), device=v.device), s + idx] = 1.0

        return v

    def backward(self, h: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Decoder-compatible backward pass: compute p(v|h).

        Args:
            h: Hidden units [batch_size, num_hidden]
            return_logits: If True, return logits instead of probabilities

        Returns:
            Visible probabilities or logits [batch_size, num_visible]
        """
        logits = self._visible_logits(h)
        if return_logits:
            return logits
        return self.visible_probs(h)

    @torch.no_grad()
    def backward_sample(self, h: torch.Tensor) -> torch.Tensor:
        """Sample visible units given hidden units."""
        return self.sample_visible(self.visible_probs(h))

    @torch.no_grad()
    def gibbs_step(self, v: torch.Tensor, sample_h: bool = True, sample_v: bool = True):
        """
        Single Gibbs sampling step: v -> h -> v'.

        Args:
            v: Current visible state
            sample_h: If True, sample h; otherwise use mean-field
            sample_v: If True, sample v'; otherwise use mean-field

        Returns:
            v_next: Next visible state
            v_prob: Visible probabilities
            h: Hidden state
            h_prob: Hidden probabilities
        """
        h_prob = self.forward(v)
        h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
        v_prob = self.visible_probs(h)
        v_next = self.sample_visible(v_prob) if sample_v else v_prob
        return v_next, v_prob, h, h_prob

    @torch.no_grad()
    def train_epoch(self, data: torch.Tensor, epoch: int, max_epochs: int, CD: int = 1):
        """
        Train RBM for one epoch using Contrastive Divergence.

        Args:
            data: Training batch [batch_size, num_visible]
            epoch: Current epoch number
            max_epochs: Total number of epochs
            CD: Number of Gibbs steps for negative phase

        Returns:
            Reconstruction loss (MSE)
        """
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = data.size(0)

        # Positive phase
        pos_h = self.forward(data)
        pos_assoc = data.T @ pos_h

        # Negative phase (CD-k)
        h = (pos_h > torch.rand_like(pos_h)).float()
        for _ in range(int(CD)):
            v_prob = self.visible_probs(h)
            v = self.sample_visible(v_prob)
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float()
        neg_assoc = v.T @ h_prob

        # Weight update with momentum
        self.W_m.mul_(mom).add_(lr * ((pos_assoc - neg_assoc) / bsz - self.weight_decay * self.W))
        self.W.add_(self.W_m)

        # Hidden bias update with optional sparsity
        self.hb_m.mul_(mom).add_(lr * (pos_h.sum(0) - h_prob.sum(0)) / bsz)
        if self.sparsity:
            Q = pos_h.mean(0)
            self.hb_m.add_(-lr * (Q - self.sparsity_factor))
        self.hid_bias.add_(self.hb_m)

        # Visible bias update
        self.vb_m.mul_(mom).add_(lr * (data.sum(0) - v.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        loss = torch.mean((data - v_prob) ** 2)
        return loss

    def _lin_schedule(self, t, t_max, start, end):
        """Linear schedule from start to end over t_max steps."""
        if t_max <= 1:
            return float(end)
        alpha = min(max(t / (t_max - 1), 0.0), 1.0)
        return float(start + (end - start) * alpha)

    def _hot_steps(self, n_steps, hot_frac):
        """Compute number of 'hot' steps (with stochastic sampling)."""
        return int(max(0, min(n_steps, round(hot_frac * n_steps))))

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
        """
        Conditional Gibbs sampling with temperature annealing.

        Reconstructs unknown parts of v while keeping known parts fixed.

        Args:
            v_known: Known visible values [batch_size, num_visible]
            known_mask: Binary mask (1 = known, 0 = unknown)
            n_steps: Number of Gibbs steps
            T0: Initial temperature
            T1: Final temperature
            sample_h_until: Sample h stochastically until this step
            sample_v_every: Sample v every N steps (0 = never)
            final_meanfield: If True, final pass uses mean-field

        Returns:
            Reconstructed visible units
        """
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)

        hot_steps = int(max(0, min(n_steps, sample_h_until)))

        for t in range(int(n_steps)):
            Tt = self._lin_schedule(t, n_steps, T0, T1)
            # Sharpen in last few steps
            if (n_steps - t) <= 3:
                Tt = min(0.9, Tt)

            h_prob = self.forward(v, T=Tt)
            h = (h_prob > torch.rand_like(h_prob)).float() if t < hot_steps else h_prob

            v_prob = self.visible_probs(h, T=Tt)
            if (t < hot_steps) and (sample_v_every > 0) and (t % sample_v_every == 0):
                v_new = self.sample_visible(v_prob)
            else:
                v_new = v_prob

            # Re-clamp known values
            v = v_new * (1 - km) + v_known * km

        # Optional final mean-field pass
        if final_meanfield:
            h_prob = self.forward(v, T=1.0)
            v = self.visible_probs(h_prob, T=1.0) * (1 - km) + v_known * km

        return v

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
        """
        Noisy mean-field inference with annealing (more robust than conditional Gibbs).

        Adds noise to logits during early steps to escape local minima.

        Args:
            v_known: Known visible values
            known_mask: Binary mask (1 = known, 0 = unknown)
            n_steps: Number of inference steps
            T0: Initial temperature
            T1: Final temperature
            sigma0: Initial noise std for logits
            hot_frac: Fraction of steps with noise
            sharpen_last: Number of final steps with extra sharpening
            T_cold_plus: Temperature for sharpening steps

        Returns:
            Reconstructed visible units
        """
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)

        hot_steps = self._hot_steps(n_steps, hot_frac)

        for t in range(int(n_steps)):
            Tt = self._lin_schedule(t, n_steps, T0, T1)
            if (n_steps - t) <= max(1, int(sharpen_last)):
                Tt = T_cold_plus
            sig_t = sigma0 * max(0.0, 1.0 - (t / max(1, n_steps - 1)))

            # h|v with noise on logits
            h_logits = (v @ self.W + self.hid_bias) / max(1e-6, Tt)
            if sig_t > 0:
                h_logits = h_logits + torch.randn_like(h_logits) * sig_t
            h_prob = torch.sigmoid(h_logits)

            # v|h with noise on logits
            v_logits = (h_prob @ self.W.T + self.vis_bias) / max(1e-6, Tt)
            if sig_t > 0:
                v_logits = v_logits + torch.randn_like(v_logits) * sig_t

            v_prob = torch.sigmoid(v_logits)
            for s, e in getattr(self, "softmax_groups", []):
                v_prob[:, s:e] = torch.softmax(v_logits[:, s:e], dim=1)

            # Optional μ-pull (class mean guidance)
            if hasattr(self, "_mu_pull") and self._mu_pull is not None:
                Dz = self._mu_pull["mu_k"].size(1)
                eta0 = float(self._mu_pull.get("eta0", 0.15))
                eta_t = eta0 * max(0.0, 1.0 - (t / max(1, n_steps - 1)))
                v_prob[:, :Dz] = (1 - eta_t) * v_prob[:, :Dz] + eta_t * self._mu_pull["mu_k"]

            v = v_prob * (1 - km) + v_known * km

        return v

    @torch.no_grad()
    def conditional_gibbs(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        n_steps: int = 30,
        sample_h: bool = False,
        sample_v: bool = False,
    ) -> torch.Tensor:
        """
        Simple conditional Gibbs sampling (without annealing).

        Args:
            v_known: Known visible values
            known_mask: Binary mask (1 = known, 0 = unknown)
            n_steps: Number of Gibbs steps
            sample_h: If True, sample h stochastically
            sample_v: If True, sample v stochastically

        Returns:
            Reconstructed visible units
        """
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
        reclamp_negative: bool = True,
        aux_lr_mult: float = 0.3,
        use_noisy_init: bool = True,
    ):
        """
        Auxiliary clamped CD training for conditional distributions.

        Used for training joint RBM with partial observations (e.g., image+label).

        Args:
            v_known: Known visible values
            known_mask: Binary mask (1 = known, 0 = unknown)
            epoch: Current epoch number
            max_epochs: Total number of epochs
            CD: Number of Gibbs steps for negative phase
            cond_init_steps: Initialization steps for positive phase
            sample_h: Sample h in negative phase
            sample_v: Sample v in negative phase
            reclamp_negative: If True, re-clamp known values in negative phase
            aux_lr_mult: Learning rate multiplier for auxiliary training
            use_noisy_init: Use noisy mean-field for initialization

        Returns:
            Reconstruction loss
        """
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = v_known.size(0)

        # Positive phase: initialize with conditional inference
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

        # Negative phase: Gibbs sampling
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

        # Updates with reduced learning rate
        wd_term = self.weight_decay * self.W
        self.W_m.mul_(mom).add_(aux_lr_mult * lr * ((pos_assoc - neg_assoc) / bsz - wd_term))
        self.W.add_(self.W_m)
        self.hb_m.mul_(mom).add_(aux_lr_mult * lr * (h_plus.sum(0) - h_neg.sum(0)) / bsz)
        self.hid_bias.add_(self.hb_m)
        self.vb_m.mul_(mom).add_(aux_lr_mult * lr * (v_plus.sum(0) - v_neg.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        return torch.mean((v_plus - v_neg) ** 2)
