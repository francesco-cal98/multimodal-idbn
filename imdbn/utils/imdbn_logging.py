from __future__ import annotations

"""
Utility logging and visualization helpers for the iMDBN model.

These functions mirror the logging-related methods previously defined
inside `src/classes/gdbn_model.py::iMDBN`. They accept the model
instance as the first argument and use its attributes (e.g.,
`image_idbn`, `joint_rbm`, `val_loader`, `wandb_run`, etc.).

Keeping these helpers here makes the iMDBN class slimmer and easier to
read while preserving the original behavior and signatures via thin
wrappers in the class.
"""

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_latent_trajectory_with_recon_panel(
    model,
    sample_idx: int = 0,
    steps: int = 40,
    tag: str = "pca_traj_with_recon",
    n_frames: int = None,
    scatter_size: int = None,
    scatter_alpha: float = None,
):
    """
    - Fit PCA(2) su tutto il validation z_img (come nei log originali).
    - Plotta il cloud + traiettoria TXT->IMG del sample.
    - Pannello a destra con GT + ricostruzioni a tappe.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    from imdbn.utils.probe_utils import compute_val_embeddings_and_features

    assert model.val_loader is not None, "val_loader mancante"

    Z_val_t, feats = compute_val_embeddings_and_features(
        model.image_idbn, upto_layer=len(model.image_idbn.layers)
    )
    if Z_val_t is None or Z_val_t.numel() == 0:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "no val embeddings"})
        return

    Z_val = Z_val_t.detach().cpu().numpy()  # [N_val, Dz]
    N_val, Dz = Z_val.shape
    sample_idx = int(max(0, min(sample_idx, N_val - 1)))

    # Vettore colori (Numerosity/N_list se presente, altrimenti Labels)
    color_vec = None
    try:
        base = model.val_loader.dataset.dataset
        indices = model.val_loader.dataset.indices
        if hasattr(base, "N_list"):
            import numpy as np

            color_vec = np.array([base.N_list[i] for i in indices], dtype=float)
    except Exception:
        pass
    if color_vec is None:
        if "labels" in feats:
            color_vec = feats["labels"].numpy()
        else:
            import numpy as np

            color_vec = np.zeros(Z_val.shape[0], dtype=float)

    # Read logging cfg (with defaults)
    cfg = getattr(model, "logging_cfg", {}) or {}
    pca_cfg = ((cfg.get("logging") or {}).get("pca_trajectory") or {})
    if n_frames is None:
        n_frames = int(pca_cfg.get("n_frames", 8))
    if scatter_size is None:
        scatter_size = int(pca_cfg.get("scatter_size", 12))
    if scatter_alpha is None:
        scatter_alpha = float(pca_cfg.get("scatter_alpha", 0.35))

    # PCA(2) fit su tutto il val (refit every call)
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z_val)  # [N_val, 2]
    z_true_2d = Z2[sample_idx : sample_idx + 1]  # [1, 2]

    # Recupera sample esatto (x_i, y_i) dal val_loader (no shuffle)
    seen = 0
    x_i = None
    y_i = None
    for imgs, lbls in model.val_loader:
        b = imgs.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        x_i = imgs[pos : pos + 1].to(model.device).view(1, -1).float()  # [1, Npix]
        y_i = lbls[pos : pos + 1].to(model.device).float()  # [1, K]
        break
    if x_i is None:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "sample not found"})
        return
    gt_class = int(y_i.argmax(dim=1).item())

    # TXT -> IMG mean-field condizionale, salva traiettoria (in PCA) e ricostruzioni
    K = model.num_labels
    V = Dz + K
    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz:] = y_i
    km[:, Dz:] = 1.0

    try:
        if hasattr(model, "z_class_mean") and model.z_class_mean is not None:
            z0 = model.z_class_mean[y_i.argmax(dim=1)]
            v_cur = v_known.clone()
            v_cur[:, :Dz] = z0
        else:
            h0 = model.joint_rbm.forward(v_known)
            v_prob0 = model.joint_rbm.visible_probs(h0)
            v_cur = v_prob0 * (1 - km) + v_known * km
    except Exception:
        h0 = model.joint_rbm.forward(v_known)
        v_prob0 = model.joint_rbm.visible_probs(h0)
        v_cur = v_prob0 * (1 - km) + v_known * km

    # Traiettoria 2D + ricostruzioni
    traj = []
    recon_imgs = []  # ciascuna [H, W] (numpy)
    x_gt = x_i.detach().cpu()
    Npix = x_gt.numel()
    side = int(round(Npix ** 0.5))
    if side * side != Npix:
        H, W = int(Npix), 1
    else:
        H = W = side

    def _vec_to_img_np(vec_1xN: torch.Tensor):
        import numpy as np

        v = vec_1xN.view(-1).detach().cpu()
        if side * side == Npix:
            img = v.view(1, H, W).clamp(0, 1)[0]
        else:
            img = v.view(1, H, W).clamp(0, 1)[0]
        return img.numpy()

    # Usa ESATTAMENTE lo stesso metodo di _cross_reconstruct per consistenza
    # Questo include: annealing, μ-pull, best-of-K
    # Tracciamo gli step intermedi salvando manualmente

    traj_points = []
    recon_points = []

    # Step 0: posizione iniziale
    z_init = v_cur[:, :Dz].detach().cpu().numpy()
    traj_points.append(z_init[0])
    img_init = model.image_idbn.decode(v_cur[:, :Dz]).detach()
    recon_points.append(_vec_to_img_np(img_init))

    # Chiamata al metodo completo come nel training
    # Generiamo l'immagine con il metodo di training
    z_img_true = model.image_idbn.represent(x_i)
    img_from_txt, _ = model._cross_reconstruct(z_img_true, y_i, steps=steps)

    # Posizione finale (dopo tutti gli step + best-of-K)
    z_final = model.image_idbn.represent(img_from_txt.view(1, -1))
    z_final_np = z_final.detach().cpu().numpy()
    traj_points.append(z_final_np[0])
    recon_points.append(_vec_to_img_np(img_from_txt))

    # Per la traiettoria usiamo interpolazione lineare tra start e end
    # (Il metodo annealed non espone gli step intermedi)
    num_frames = min(int(steps / 5), 8)
    for i in range(1, num_frames):
        alpha = i / num_frames
        z_interp = (1 - alpha) * z_init + alpha * z_final_np
        traj_points.append(z_interp[0])
        img_interp = model.image_idbn.decode(torch.from_numpy(z_interp).float().to(model.device))
        recon_points.append(_vec_to_img_np(img_interp))

    # Converti in array per PCA
    import numpy as np
    traj = np.array([pca.transform(z.reshape(1, -1))[0] for z in traj_points])
    recon_imgs = recon_points

    # Seleziona n_frames tappe rappresentative (oltre alla GT)
    if n_frames < 2:
        n_frames = 2
    sel_idx = np.unique(np.linspace(0, len(recon_imgs) - 1, n_frames, dtype=int)).tolist()

    panel_imgs = []
    panel_titles = []
    panel_imgs.append(_vec_to_img_np(x_gt))
    panel_titles.append("GT")
    for si in sel_idx:
        panel_imgs.append(recon_imgs[si])
        panel_titles.append(f"step {si}")

    import math
    import matplotlib.pyplot as plt
    import wandb

    n_tiles = len(panel_imgs)
    rows = 2
    cols = math.ceil(n_tiles / rows)

    fig = plt.figure(figsize=(8 + cols * 2.2, max(6, rows * 2.2)))
    gs = fig.add_gridspec(nrows=rows, ncols=cols + 4)

    ax0 = fig.add_subplot(gs[:, :4])
    sc = ax0.scatter(
        Z2[:, 0], Z2[:, 1], c=color_vec, cmap="viridis", s=scatter_size, alpha=scatter_alpha
    )
    ax0.scatter(
        z_true_2d[0, 0], z_true_2d[0, 1], s=80, marker="*", c="k", edgecolor="w", linewidths=0.8,
        label=f"sample GT (class={gt_class})", zorder=3,
    )
    ax0.scatter(
        traj[0, 0], traj[0, 1], s=50, marker="D", c="red", edgecolor="k", linewidths=0.5,
        label="start catena", zorder=3,
    )
    ax0.plot(traj[:, 0], traj[:, 1], linewidth=1.6, marker="o", markersize=3, c="red", label="traiettoria", zorder=2)
    for t in range(0, len(traj), max(1, len(traj) // 10)):
        ax0.text(traj[t, 0], traj[t, 1], str(t), fontsize=7, color="red")

    ax0.set_title(f"PCA z_img — sample {sample_idx} (class={gt_class}) — steps={steps}")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("Numerosity / N_list (fallback: Labels)")
    ax0.legend(loc="best")

    right_gs = gs[:, 4:].subgridspec(nrows=rows, ncols=cols)
    for k, img in enumerate(panel_imgs):
        r = k // cols
        c = k % cols
        ax = fig.add_subplot(right_gs[r, c])
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(panel_titles[k], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if model.wandb_run:
        model.wandb_run.log({f"{tag}/plot": wandb.Image(fig)})
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def log_pca3_trajectory(model, sample_idx: int, steps: int = 40, tag: str = "pca3_traj"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb

    from imdbn.utils.probe_utils import compute_val_embeddings_and_features

    Z_val_t, _ = compute_val_embeddings_and_features(
        model.image_idbn, upto_layer=len(model.image_idbn.layers)
    )
    if Z_val_t is None or Z_val_t.numel() == 0:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "no val embeddings"})
        return
    Z_val = Z_val_t.cpu().numpy()

    Dz = model.Dz_img
    K = model.num_labels
    V = Dz + K

    seen, x_i, y_i = 0, None, None
    for imgs, lbls in model.val_loader:
        b = imgs.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        x_i = imgs[pos : pos + 1].to(model.device).view(1, -1).float()
        y_i = lbls[pos : pos + 1].to(model.device).float()
        break

    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz:] = y_i
    km[:, Dz:] = 1.0
    if hasattr(model, "z_class_mean") and model.z_class_mean is not None:
        z0 = model.z_class_mean[y_i.argmax(1)]
        v_cur = v_known.clone()
        v_cur[:, :Dz] = z0
    else:
        h0 = model.joint_rbm.forward(v_known)
        v_prob0 = model.joint_rbm.visible_probs(h0)
        v_cur = v_prob0 * (1 - km) + v_known * km

    zs = [v_cur[:, :Dz].detach().cpu().numpy()]
    for step_i in range(int(steps)):
        h_prob = model.joint_rbm.forward(v_cur)
        h_sample = torch.bernoulli(h_prob)
        v_logits = h_sample @ model.joint_rbm.W.T + model.joint_rbm.vis_bias
        v_prob = torch.sigmoid(v_logits)
        for s, e in getattr(model.joint_rbm, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(v_logits[:, s:e], dim=1)
        v_cur = v_prob * (1 - km) + v_known * km
        zs.append(v_cur[:, :Dz].detach().cpu().numpy())
    Z_traj = np.vstack(zs)

    # Refit PCA on every call to mirror 2D behavior
    p3 = PCA(n_components=3).fit(Z_val)
    Z3 = p3.transform(Z_val)
    T3 = p3.transform(Z_traj)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=6, alpha=0.15)
    ax.plot(T3[:, 0], T3[:, 1], T3[:, 2], c="r", linewidth=1.2)
    ax.set_title("PCA-3 trajectory")
    fig.tight_layout()
    if model.wandb_run:
        model.wandb_run.log({f"{tag}/pca3": wandb.Image(fig)})
        plt.close(fig)


@torch.no_grad()
def log_pca3_trajectory_with_recon_panel(
    model,
    sample_idx: int = 0,
    steps: int = 40,
    tag: str = "pca3_traj_with_recon",
    n_frames: int = None,
    scatter_size: int = None,
    scatter_alpha: float = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
):
    """
    3D PCA scatter + trajectory rendered similar to the 2D version,
    with a recon panel on the right. PCA is refit on validation embeddings every call.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from imdbn.utils.probe_utils import compute_val_embeddings_and_features

    # Config defaults
    cfg = getattr(model, "logging_cfg", {}) or {}
    pca_cfg = ((cfg.get("logging") or {}).get("pca_trajectory") or {})
    p3_cfg = (pca_cfg.get("three_d") or {})
    if n_frames is None:
        n_frames = int(pca_cfg.get("n_frames", 8))
    if scatter_size is None:
        scatter_size = int(p3_cfg.get("scatter_size", 6))
    if scatter_alpha is None:
        scatter_alpha = float(p3_cfg.get("scatter_alpha", 0.15))
    if elev is None:
        elev = float(p3_cfg.get("elev", 20))
    if azim is None:
        azim = float(p3_cfg.get("azim", 35))

    assert model.val_loader is not None, "val_loader mancante"

    Z_val_t, feats = compute_val_embeddings_and_features(
        model.image_idbn, upto_layer=len(model.image_idbn.layers)
    )
    if Z_val_t is None or Z_val_t.numel() == 0:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "no val embeddings"})
        return

    Z_val = Z_val_t.detach().cpu().numpy()
    N_val, Dz = Z_val.shape
    sample_idx = int(max(0, min(sample_idx, N_val - 1)))

    # Color vector
    color_vec = None
    try:
        base = model.val_loader.dataset.dataset
        indices = model.val_loader.dataset.indices
        if hasattr(base, "N_list"):
            color_vec = np.array([base.N_list[i] for i in indices], dtype=float)
    except Exception:
        pass
    if color_vec is None:
        if "labels" in feats:
            color_vec = feats["labels"].numpy()
        else:
            color_vec = np.zeros(Z_val.shape[0], dtype=float)

    # Refit PCA(3)
    pca3 = PCA(n_components=3)
    Z3 = pca3.fit_transform(Z_val)
    z_true_3d = Z3[sample_idx : sample_idx + 1]

    # Fetch sample (x_i, y_i) from val_loader
    seen = 0
    x_i = None
    y_i = None
    for imgs, lbls in model.val_loader:
        b = imgs.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        x_i = imgs[pos : pos + 1].to(model.device).view(1, -1).float()
        y_i = lbls[pos : pos + 1].to(model.device).float()
        break
    if x_i is None:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "sample not found"})
        return
    gt_class = int(y_i.argmax(dim=1).item())

    # Build TXT->IMG mean-field trajectory
    K = model.num_labels
    V = Dz + K
    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz:] = y_i
    km[:, Dz:] = 1.0

    try:
        if hasattr(model, "z_class_mean") and model.z_class_mean is not None:
            z0 = model.z_class_mean[y_i.argmax(dim=1)]
            v_cur = v_known.clone()
            v_cur[:, :Dz] = z0
        else:
            h0 = model.joint_rbm.forward(v_known)
            v_prob0 = model.joint_rbm.visible_probs(h0)
            v_cur = v_prob0 * (1 - km) + v_known * km
    except Exception:
        h0 = model.joint_rbm.forward(v_known)
        v_prob0 = model.joint_rbm.visible_probs(h0)
        v_cur = v_prob0 * (1 - km) + v_known * km

    # Prepare recon helper
    x_gt = x_i.detach().cpu()
    Npix = x_gt.numel()
    side = int(round(Npix ** 0.5))
    if side * side != Npix:
        H, W = Npix, 1
    else:
        H = W = side

    def _vec_to_img_np(vec_1xN: torch.Tensor):
        v = vec_1xN.view(-1).detach().cpu()
        return v.view(1, H, W).clamp(0, 1)[0].numpy()

    # Accumulate trajectory + reconstructions
    traj3 = []
    recon_imgs = []
    z_init = v_cur[:, :Dz].detach().cpu().numpy()
    traj3.append(pca3.transform(z_init)[0])
    img0 = model.image_idbn.decode(v_cur[:, :Dz]).detach()
    recon_imgs.append(_vec_to_img_np(img0))
    for step_i in range(int(steps)):
        h_prob = model.joint_rbm.forward(v_cur)
        h_sample = torch.bernoulli(h_prob)
        v_logits = h_sample @ model.joint_rbm.W.T + model.joint_rbm.vis_bias
        v_prob = torch.sigmoid(v_logits)
        for s, e in getattr(model.joint_rbm, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(v_logits[:, s:e], dim=1)
        v_cur = v_prob * (1 - km) + v_known * km
        z_t = v_cur[:, :Dz].detach().cpu().numpy()
        traj3.append(pca3.transform(z_t)[0])
        img_t = model.image_idbn.decode(v_cur[:, :Dz]).detach()
        recon_imgs.append(_vec_to_img_np(img_t))

    traj3 = np.stack(traj3, axis=0)

    # Panel frame selection
    if n_frames < 2:
        n_frames = 2
    sel_idx = np.unique(np.linspace(0, len(recon_imgs) - 1, n_frames, dtype=int)).tolist()

    panel_imgs = []
    panel_titles = []
    panel_imgs.append(_vec_to_img_np(x_gt))
    panel_titles.append("GT")
    for si in sel_idx:
        panel_imgs.append(recon_imgs[si])
        panel_titles.append(f"step {si}")

    import math
    import wandb

    n_tiles = len(panel_imgs)
    rows = 2
    cols = math.ceil(n_tiles / rows)

    fig = plt.figure(figsize=(8 + cols * 2.2, max(6, rows * 2.2)))
    gs = fig.add_gridspec(nrows=rows, ncols=cols + 4)

    ax0 = fig.add_subplot(gs[:, :4], projection="3d")
    sc = ax0.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=color_vec, cmap="viridis", s=scatter_size, alpha=scatter_alpha)
    ax0.scatter(
        z_true_3d[0, 0], z_true_3d[0, 1], z_true_3d[0, 2], s=80, marker="*", c="k", edgecolor="w", linewidths=0.8,
        label=f"sample GT (class={gt_class})", zorder=3,
    )
    ax0.scatter(
        traj3[0, 0], traj3[0, 1], traj3[0, 2], s=50, marker="D", c="red", edgecolor="k", linewidths=0.5,
        label="start catena", zorder=3,
    )
    ax0.plot(traj3[:, 0], traj3[:, 1], traj3[:, 2], linewidth=1.6, marker="o", markersize=3, c="red", label="traiettoria", zorder=2)
    try:
        ax0.view_init(elev=elev, azim=azim)
    except Exception:
        pass
    ax0.set_title(f"PCA-3 z_img — sample {sample_idx} (class={gt_class}) — steps={steps}")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_zlabel("PC3")
    cb = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("Numerosity / N_list (fallback: Labels)")
    ax0.legend(loc="best")

    right_gs = gs[:, 4:].subgridspec(nrows=rows, ncols=cols)
    for k, img in enumerate(panel_imgs):
        r = k // cols
        c = k % cols
        ax = fig.add_subplot(right_gs[r, c])
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(panel_titles[k], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if model.wandb_run:
        model.wandb_run.log({f"{tag}/plot": wandb.Image(fig)})
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def panel_with_gt_and_neighbors(
    model,
    panel_title: str,
    gt_img: torch.Tensor,
    neighbor_imgs: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    tag_key: str,
):
    """
    Figura con GT come primo riquadro + k vicini con caption:
    - GT: 'Ground Truth'
    - Vicino r: 'rank r | idx i | score s | label L'
    """
    import math
    import matplotlib.pyplot as plt
    import wandb

    def _to_chw(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu()
        if x.ndim == 2:
            side = int(round(x.size(1) ** 0.5))
            x = x.view(x.size(0), 1, side, side)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    gt = _to_chw(gt_img)  # [1,1,H,W]
    nbr = _to_chw(neighbor_imgs)  # [k,1,H,W]
    k = nbr.size(0)
    y_idx_all = model._Y_bank.argmax(1)  # [N]
    labels = [int(y_idx_all[i].item()) for i in neighbor_indices.tolist()]

    rows = 2
    cols = math.ceil((k + 1) / rows)
    fig = plt.figure(figsize=(cols * 2.6, rows * 2.8))
    fig.suptitle(panel_title, fontsize=12)

    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(gt[0, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Ground Truth", fontsize=10)
    ax.axis("off")

    for r in range(k):
        ax = fig.add_subplot(rows, cols, r + 2)
        ax.imshow(nbr[r, 0], cmap="gray", vmin=0, vmax=1)
        idx_i = int(neighbor_indices[r].item())
        sc_i = float(neighbor_scores[r].item())
        lab_i = labels[r]
        ax.set_title(
            f"rank {r} | idx {idx_i}\nscore {sc_i:.4f} | label {lab_i}", fontsize=8
        )
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if model.wandb_run:
        model.wandb_run.log({tag_key: wandb.Image(fig)})
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def panel_gt_vs_decode_neighbors(model, panel_title: str, neighbor_indices: torch.Tensor, tag_key: str):
    import matplotlib.pyplot as plt
    import wandb

    pick = neighbor_indices.to(torch.long)
    X = model._X_bank[pick]  # GT
    Z = model._Z_bank[pick].to(model.device).float()
    rec = model.image_idbn.decode(Z).detach().cpu()

    def _to_chw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            side = int(round(x.size(1) ** 0.5))
            return x.view(x.size(0), 1, side, side)
        elif x.ndim == 4:
            return x
        else:
            return x.unsqueeze(1)

    X = _to_chw(X)
    rec = _to_chw(rec)
    k = X.size(0)
    fig = plt.figure(figsize=(2 * 2.4, k * 2.2))
    fig.suptitle(panel_title, fontsize=12)
    for r in range(k):
        ax = fig.add_subplot(k, 2, 2 * r + 1)
        ax.imshow(X[r, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("GT", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(k, 2, 2 * r + 2)
        ax.imshow(rec[r, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Decode(z)", fontsize=9)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if model.wandb_run:
        model.wandb_run.log({tag_key: wandb.Image(fig)})
        plt.close(fig)


@torch.no_grad()
def ensure_val_bank(model) -> None:
    """Build and cache (Z, X, Y, H) banks for validation set on the model."""
    if hasattr(model, "_Z_bank"):
        return
    Z_list, X_list, Y_list, H_list = [], [], [], []
    for imgs, lbls in model.val_loader:
        z = model.image_idbn.represent(imgs.view(imgs.size(0), -1).float().to(model.device))
        Z_list.append(z.cpu())
        X_list.append(imgs.cpu())
        Y_list.append(lbls.cpu())
        flat = imgs.view(imgs.size(0), -1).float().cpu()
        h = torch.stack([flat.sum(1), (flat ** 2).sum(1)], dim=1)
        H_list.append(h)
    model._Z_bank = torch.cat(Z_list, 0)
    model._X_bank = torch.cat(X_list, 0)
    model._Y_bank = torch.cat(Y_list, 0)
    model._H_bank = torch.cat(H_list, 0)


@torch.no_grad()
def find_first_val_index_with_label(model, k: int) -> int:
    idx = 0
    for _, lbls in model.val_loader:
        y = lbls.argmax(1)
        for j in range(y.size(0)):
            if int(y[j].item()) == int(k):
                return idx + j
        idx += y.size(0)
    return -1


@torch.no_grad()
def topk_similar_in_latent(model, z_query: torch.Tensor, k: int = 8, metric: str = "cosine"):
    """Search top-k neighbors in validation latent bank.

    Returns (indices, scores), both Float/Long tensors on CPU.
    """
    assert hasattr(model, "_Z_bank"), "Call ensure_val_bank() first."
    Z = model._Z_bank  # [N, Dz] CPU
    if metric == "cosine":
        Zn = F.normalize(Z, dim=1)
        zq = F.normalize(z_query.detach().cpu(), dim=1)
        scores = zq @ Zn.T
    elif metric in ("ip", "inner"):
        zq = z_query.detach().cpu()
        scores = zq @ Z.T
    else:
        zq = z_query.detach().cpu()
        a2 = (zq ** 2).sum(1, keepdim=True)
        b2 = (Z ** 2).sum(1).unsqueeze(0)
        ip = zq @ Z.T
        scores = -(a2 + b2 - 2 * ip)
    topv, topi = torch.topk(scores, k=min(k, Z.size(0)), dim=1)
    return topi, topv


@torch.no_grad()
def log_vecdb_neighbors_for_traj(
    model,
    sample_idx: int = 0,
    steps: Optional[int] = None,
    k: int = 8,
    metric: str = "cosine",
    tag: str = "vecdb",
    also_l2: bool = True,
    dedup: str = "index",
    exclude_self: bool = True,
):
    """
    Per un sample del validation set:
    - genera la traiettoria TXT→IMG (mean-field): z_true, z0, zT,
    - fa similarity search nel Dz completo con metrica scelta,
    - deduplica i vicini ed esclude (opzionale) il sample stesso,
    - logga griglie classiche + pannelli con Ground Truth,
    - (opzionale) pannello L2 per zT e GT vs Decode(z) dei vicini.
    """
    import numpy as np
    import torchvision.utils as vutils
    import wandb

    ensure_val_bank(model)
    Z_bank, X_bank, Y_bank, H_bank = model._Z_bank, model._X_bank, model._Y_bank, model._H_bank
    N, Dz = Z_bank.shape

    # recupera (x_i, y_i)
    seen = 0
    x_i = None
    y_i = None
    for imgs, lbls in model.val_loader:
        b = imgs.size(0)
        if seen + b <= sample_idx:
            seen += b
            continue
        pos = sample_idx - seen
        x_i = imgs[pos : pos + 1].to(model.device).view(1, -1).float()
        y_i = lbls[pos : pos + 1].to(model.device).float()
        break
    if x_i is None:
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/warn": "sample_idx out of range"})
        return

    # traiettoria TXT→IMG
    Dz_img = model.Dz_img
    K = model.num_labels
    V = Dz_img + K
    v_known = torch.zeros(1, V, device=model.device)
    km = torch.zeros_like(v_known)
    v_known[:, Dz_img:] = y_i
    km[:, Dz_img:] = 1.0
    if hasattr(model, "z_class_mean") and model.z_class_mean is not None:
        z0_init = model.z_class_mean[y_i.argmax(1)]
        v_cur = v_known.clone()
        v_cur[:, :Dz_img] = z0_init
    else:
        h0 = model.joint_rbm.forward(v_known)
        v_prob0 = model.joint_rbm.visible_probs(h0)
        v_cur = v_prob0 * (1 - km) + v_known * km

    T = int(model.cross_steps if steps is None else steps)
    zs = [v_cur[:, :Dz_img].detach().cpu().numpy()]
    for step_i in range(T):
        h_prob = model.joint_rbm.forward(v_cur)
        h_sample = torch.bernoulli(h_prob)
        v_logits = h_sample @ model.joint_rbm.W.T + model.joint_rbm.vis_bias
        v_prob = torch.sigmoid(v_logits)
        for s, e in getattr(model.joint_rbm, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(v_logits[:, s:e], dim=1)
        v_cur = v_prob * (1 - km) + v_known * km
        zs.append(v_cur[:, :Dz_img].detach().cpu().numpy())
    Z_traj = np.vstack(zs)  # [T+1, Dz]

    z_true = model.image_idbn.represent(x_i).to(model.device)  # [1,Dz]
    z0 = torch.from_numpy(Z_traj[:1]).to(model.device).float()  # [1,Dz]
    zT = torch.from_numpy(Z_traj[-1:]).to(model.device).float()  # [1,Dz]

    def score(zq: torch.Tensor, met: str) -> torch.Tensor:
        if met == "cosine":
            return F.normalize(zq.detach().cpu(), 1) @ F.normalize(Z_bank, 1).T
        elif met in ("inner", "ip"):
            return zq.detach().cpu() @ Z_bank.T
        else:
            zq = zq.detach().cpu()
            a2 = (zq ** 2).sum(1, keepdim=True)
            b2 = (Z_bank ** 2).sum(1).unsqueeze(0)
            ip = zq @ Z_bank.T
            return -(a2 + b2 - 2 * ip)

    def topk_dedup(zq: torch.Tensor, met: str, k: int, name: str):
        scores = score(zq, met).squeeze(0)  # [N]
        vals, idx = torch.sort(scores, descending=True)
        ids = idx.tolist()
        vs = vals.tolist()
        picked_ids, picked_vs = [], []
        seen_idx = set()
        seen_hash = set()
        for i, v in zip(ids, vs):
            if exclude_self and i == sample_idx:
                continue
            if dedup == "index":
                if i in seen_idx:
                    continue
                seen_idx.add(i)
            elif dedup == "image":
                key = (float(H_bank[i, 0].item()), float(H_bank[i, 1].item()))
                if key in seen_hash:
                    continue
                seen_hash.add(key)
            picked_ids.append(i)
            picked_vs.append(v)
            if len(picked_ids) >= k:
                break
        return torch.tensor(picked_ids, dtype=torch.long).unsqueeze(0), torch.tensor(picked_vs).unsqueeze(0)

    idx_true, sc_true = topk_dedup(z_true, metric, k, name="z_true")
    idx_z0, sc_z0 = topk_dedup(z0, metric, k, name="z0")
    idx_zT, sc_zT = topk_dedup(zT, metric, k, name="zT")

    # Griglie classiche con le immagini dei vicini
    def _mkgrid(indices: torch.Tensor):
        X = X_bank[indices[0].to(torch.long)]
        if X.ndim == 2:
            Npix = X.size(1)
            side = int(round(Npix ** 0.5))
            X = X.view(X.size(0), 1, side, side)
        return vutils.make_grid(X, nrow=min(4, X.size(0)))

    for name, idx in [("z_true", idx_true), ("z0", idx_z0), ("zT", idx_zT)]:
        grid = _mkgrid(idx)
        if model.wandb_run:
            model.wandb_run.log({f"{tag}/knn_{name}": wandb.Image(grid.permute(1, 2, 0).numpy())})

    # Pannelli con Ground Truth per z_true, z0, zT
    gt_img = x_i.detach().cpu()
    imgs_true = X_bank[idx_true[0]]
    imgs_z0 = X_bank[idx_z0[0]]
    imgs_zT = X_bank[idx_zT[0]]
    panel_with_gt_and_neighbors(
        model,
        panel_title="Neighbors of z_true with GT",
        gt_img=gt_img,
        neighbor_imgs=imgs_true,
        neighbor_indices=idx_true[0],
        neighbor_scores=sc_true[0],
        tag_key=f"{tag}/knn_true_with_gt",
    )
    panel_with_gt_and_neighbors(
        model,
        panel_title="Neighbors of z0 with GT",
        gt_img=gt_img,
        neighbor_imgs=imgs_z0,
        neighbor_indices=idx_z0[0],
        neighbor_scores=sc_z0[0],
        tag_key=f"{tag}/knn_z0_with_gt",
    )
    panel_with_gt_and_neighbors(
        model,
        panel_title="Neighbors of zT with GT",
        gt_img=gt_img,
        neighbor_imgs=imgs_zT,
        neighbor_indices=idx_zT[0],
        neighbor_scores=sc_zT[0],
        tag_key=f"{tag}/knn_zT_with_gt",
    )

    # (opzionale) pannello L2 per zT
    if also_l2:
        idx_zT_l2, sc_zT_l2 = topk_dedup(zT, "l2", k, name="zT_l2")
        imgs_zT_l2 = X_bank[idx_zT_l2[0]]
        panel_with_gt_and_neighbors(
            model,
            panel_title="Neighbors of zT (L2) with GT",
            gt_img=gt_img,
            neighbor_imgs=imgs_zT_l2,
            neighbor_indices=idx_zT_l2[0],
            neighbor_scores=sc_zT_l2[0],
            tag_key=f"{tag}/knn_zT_l2_with_gt",
        )

    # GT vs Decode(z) per i vicini di zT (cosine)
    panel_gt_vs_decode_neighbors(
        model,
        panel_title="Neighbors of zT — GT vs Decode(z)",
        neighbor_indices=idx_zT[0],
        tag_key=f"{tag}/knn_zT_gt_vs_decode",
    )


@torch.no_grad()
def log_neighbors_images(model, indices: torch.Tensor, tag: str):
    import torchvision.utils as vutils
    import wandb

    X = model._X_bank
    pick = indices[0].to(torch.long)
    sel = X[pick]
    if sel.ndim == 2:
        Npix = sel.size(1)
        side = int(round(Npix ** 0.5))
        sel = sel.view(sel.size(0), 1, side, side)
    grid = vutils.make_grid(sel, nrow=min(4, sel.size(0)))
    if model.wandb_run:
        model.wandb_run.log({tag: wandb.Image(grid.permute(1, 2, 0).numpy())})


@torch.no_grad()
def log_joint_auto_recon(model, epoch: int, num: int = 8):
    import torchvision.utils as vutils
    import wandb

    if model.wandb_run is None or model.validation_images is None or model.validation_labels is None:
        return

    imgs = model.validation_images[:num]
    lbls = model.validation_labels[:num]

    # forward joint
    z_top = model.image_idbn.represent(imgs.view(imgs.size(0), -1))
    v = torch.cat([z_top, lbls.to(model.device).float()], dim=1)
    h = model.joint_rbm.forward(v)
    v_recon = model.joint_rbm.backward(h)  # [B, Dz + K]
    Dz = model.Dz_img
    z_img_hat = v_recon[:, :Dz]
    y_hat = v_recon[:, Dz:]

    # decodifica immagine dal top
    rec_img = model.image_idbn.decode(z_img_hat).clamp(0, 1)

    # rendi entrambe le immagini 4D (B, C, H, W)
    if imgs.ndim == 4:
        B, C, H, W = imgs.shape
        imgs4 = imgs
    else:
        B = imgs.size(0)
        N = imgs.size(1)
        side = int(round(N ** 0.5))
        if side * side != N:
            side = int(N)
        C, H, W = 1, side, side
        imgs4 = imgs.view(B, C, H, W)

    rec4 = rec_img.view(B, C, H, W)

    # grid GT vs Joint
    pair = torch.stack([imgs4.cpu(), rec4.cpu()], dim=1).view(-1, C, H, W)
    grid = vutils.make_grid(pair, nrow=2)
    model.wandb_run.log({"auto_recon/gt_vs_joint": wandb.Image(grid.permute(1, 2, 0).numpy()), "epoch": epoch})

    # metriche testo dal joint
    gt = lbls.argmax(dim=1)
    pred = y_hat.argmax(dim=1)
    top1 = (pred == gt).float().mean().item()
    model.wandb_run.log({"auto_recon/text_top1": top1, "epoch": epoch})

    text_bce = F.binary_cross_entropy(y_hat.clamp(1e-6, 1 - 1e-6), lbls.float()).item()
    model.wandb_run.log({"auto_recon/text_bce": text_bce, "epoch": epoch})

    # MSE immagine (usa viste flatten coerenti)
    mse = F.mse_loss(imgs4.view(B, -1).to(model.device), rec4.view(B, -1).to(model.device)).item()
    model.wandb_run.log({"auto_recon/image_mse": mse, "epoch": epoch})
