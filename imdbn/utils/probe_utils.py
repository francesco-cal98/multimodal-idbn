# probe_utils.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd  # <-- per DataFrame "stile Excel"
import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# Embeddings + features
# -------------------------
@torch.no_grad()
def compute_val_embeddings_and_features(model, upto_layer: int | None = None) -> tuple[torch.Tensor, dict]:
    assert model.val_loader is not None, "val_loader is None."

    embeds = []
    for batch_data, batch_labels in model.val_loader:
        x = (batch_data.to(model.device) if not getattr(model, "text_flag", False) else batch_labels.to(model.device))
        x = x.view(x.size(0), -1).float()
        z = model.represent(x) if upto_layer is None else model.represent(x, upto_layer=upto_layer)
        embeds.append(z.detach().cpu())
    E = torch.cat(embeds, dim=0)  # [N, D]

    # --- helper per trovare la chiave ignorando spazi/maiuscole ---
    def _get_feat(d: dict, *candidates):
        norm = {k.lower().replace(" ", "").replace("_", ""): k for k in d.keys()}
        for c in candidates:
            key = norm.get(c.lower().replace(" ", "").replace("_", ""))
            if key is not None:
                return d[key]
        return None

    feats_src = getattr(model, "features", None)
    if feats_src is None:
        raise RuntimeError("model.features is required")

    cum_area_t = _get_feat(feats_src, "Cumulative Area", "cum_area")
    chull_t    = _get_feat(feats_src, "Convex Hull", "convex_hull", "convexhull")
    labels_t   = _get_feat(feats_src, "Labels", "labels")
    density_t  = _get_feat(feats_src, "Density", "density")

    # --- normalizza forme/dtype ---
    def _to_1d_float(t):
        if t is None:
            return None
        t = torch.as_tensor(t)
        # se è one-hot [N,K] → indici 0..K-1
        if t.ndim == 2:
            t = torch.argmax(t, dim=1)
        return t.view(-1).to(torch.float32).cpu()

    cum_area = _to_1d_float(cum_area_t)
    chull    = _to_1d_float(chull_t)
    labels   = _to_1d_float(labels_t)
    density  = _to_1d_float(density_t)

    n = E.size(0)
    def _check(name, v):
        if v is None:
            return False
        if v.numel() != n:
            # taglia o ripete per sicurezza, ma meglio segnalare
            raise RuntimeError(f"Feature '{name}' length mismatch: {v.numel()} vs embeddings {n}.")
        return True

    feats = {}
    if _check("cum_area", cum_area): feats["cum_area"] = cum_area
    if _check("convex_hull", chull): feats["convex_hull"] = chull
    if _check("labels", labels):     feats["labels"] = labels
    if density is not None and _check("density", density): feats["density"] = density

    return E, feats

@torch.no_grad()
@torch.no_grad()
def compute_joint_embeddings_and_features(model) -> tuple[torch.Tensor, dict]:
    assert model.val_loader is not None, "val_loader is None."

    embeds = []
    for img_data, labels in model.val_loader:
        z = model.represent((img_data.to(model.device), labels.to(model.device)))
        embeds.append(z.detach().cpu())
    if not embeds:
        return torch.empty(0), {}

    E = torch.cat(embeds, dim=0)

    feats_src = getattr(model, "features", None)
    if feats_src is None:
        raise RuntimeError("model.features is required")

    def _get_feat(d: dict, *candidates):
        norm = {k.lower().replace(" ", "").replace("_", ""): k for k in d.keys()}
        for c in candidates:
            key = norm.get(c.lower().replace(" ", "").replace("_", ""))
            if key is not None:
                return d[key]
        return None

    def _to_1d_float(t):
        if t is None:
            return None
        t = torch.as_tensor(t)
        if t.ndim == 2:  # one-hot
            t = torch.argmax(t, dim=1)
        return t.view(-1).to(torch.float32).cpu()

    cum_area = _to_1d_float(_get_feat(feats_src, "Cumulative Area", "cum_area"))
    chull    = _to_1d_float(_get_feat(feats_src, "Convex Hull", "convex_hull", "convexhull"))
    labels   = _to_1d_float(_get_feat(feats_src, "Labels", "labels"))
    density  = _to_1d_float(_get_feat(feats_src, "Density", "density"))

    n = E.size(0)
    def _check(name, v):
        if v is None:
            return False
        if v.numel() != n:
            raise RuntimeError(f"Feature '{name}' length mismatch: {v.numel()} vs embeddings {n}.")
        return True

    feats = {}
    if _check("cum_area", cum_area): feats["cum_area"] = cum_area
    if _check("convex_hull", chull): feats["convex_hull"] = chull
    if _check("labels", labels):     feats["labels"] = labels
    if density is not None and _check("density", density): feats["density"] = density

    return E, feats


# -------------------------
# Binning + split
# -------------------------
def make_bin_labels(values: torch.Tensor, n_bins: int = 5):
    """
    Binning per quantili:
      - ritorna labels ∈ {0..n_bins-1}
      - ritorna anche edges (tensor n_bins+1) per logging/debug
    Gestisce edge uguali aggiungendo un jitter minimo.
    """
    qs = torch.linspace(0, 1, steps=n_bins + 1)
    edges = torch.quantile(values, qs, interpolation='linear')
    # evita edge identici (dati discreti o con molti pari)
    for k in range(1, len(edges)):
        if edges[k] <= edges[k-1]:
            edges[k] = edges[k-1] + 1e-6
    inner = edges[1:-1]
    labels = torch.bucketize(values, inner, right=False)  # 0..n_bins-1
    return labels, edges


def _format_bin_names(edges: torch.Tensor, precision: int = 4):
    """
    Ritorna i nomi leggibili dei bin come 'low-high' (stringhe),
    usando gli estremi degli edges (len = n_bins+1).
    """
    e = edges.detach().cpu().numpy().astype(float)
    fmt = lambda v: f"{v:.{precision}f}".rstrip('0').rstrip('.')  # evita zeri inutili
    names = [f"{fmt(e[i])}-{fmt(e[i+1])}" for i in range(len(e)-1)]
    return names  # len = n_bins


def stratified_split(labels: torch.Tensor, test_size: float = 0.2, rng_seed: int = 42):
    """
    Split stratificato per classe/bin usando TUTTI i dati.
    Ritorna: train_idx, test_idx (liste di int).
    """
    rng = random.Random(rng_seed)
    train_idx, test_idx = [], []
    classes = torch.unique(labels).tolist()
    for c in classes:
        idxs = (labels == c).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 1:
            test_idx.extend(idxs)
            continue
        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)  # lascia almeno 1 in train
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return train_idx, test_idx


# -------------------------
# Linear classifier (probe) con Early Stopping (NO logging per-step)
# -------------------------
def train_linear_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    device:  torch.device,
    n_classes: int,
    max_steps: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 20,
    min_delta: float = 0.0,
):
    """
    Linear probe full-batch: nn.Linear(D, n_classes) + CrossEntropy.
    Early stopping su validation loss con 'patience' e 'min_delta'.
    Non effettua alcun logging per step.

    Ritorna:
      acc_val_best, y_true (val), y_pred (val)
    """
    D = X_train.shape[1]
    model = nn.Linear(D, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xva = torch.tensor(X_val,   dtype=torch.float32, device=device)
    yva = torch.tensor(y_val,   dtype=torch.long,   device=device)

    best_loss = float("inf")
    best_state = None

    model.train()
    for _ in range(max_steps):
        # train
        opt.zero_grad()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()

        # val
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva)
            v_loss = F.cross_entropy(v_logits, yva).item()
        model.train()

        # early stopping su v_loss
        if v_loss < best_loss - min_delta:
            best_loss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # eval finale con best_state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xva)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == yva).float().mean().item()

    return acc, yva.detach().cpu().tolist(), preds.detach().cpu().tolist()


# -------------------------
# Confusion matrix + logging "Excel-like"
# -------------------------
def _confusion_df(y_true, y_pred, n_classes: int, bin_names: list[str]) -> pd.DataFrame:
    """
    Crea una confusion matrix come DataFrame (righe=true, colonne=pred),
    con intestazioni leggibili (bin_names).
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    df = pd.DataFrame(cm, index=bin_names, columns=bin_names)
    df.index.name = "True"
    df.columns.name = "Pred"
    return df


def _save_confusion_csv(df: pd.DataFrame, model, metric_name: str, epoch: int) -> str:
    """
    Salva la confusion matrix in CSV dentro model.arch_dir e ritorna il path.
    """
    os.makedirs(model.arch_dir, exist_ok=True)
    path = os.path.join(model.arch_dir, f"probe_{metric_name}_confusion_epoch{epoch}.csv")
    df.to_csv(path)
    return path


def _log_confusion_table_wandb(wandb_run, df: pd.DataFrame, metric_name: str, epoch: int):
    """
    Logga la confusion matrix come tabella W&B (leggibile).
    """
    if not wandb_run or wandb is None:
        return
    try:
        table = wandb.Table(dataframe=df)
        wandb_run.log({f"probe/{metric_name}/confusion_table": table, "epoch": epoch})
    except Exception:
        # fallback minimale: logga il dict
        wandb_run.log({f"probe/{metric_name}/confusion_dict": df.to_dict(), "epoch": epoch})


def _log_accuracy_wandb(wandb_run, metric_name: str, acc: float, epoch: int):
    if not wandb_run or wandb is None:
        return
    wandb_run.log({f"probe/{metric_name}/acc": acc, "epoch": epoch})


def _log_bin_edges_wandb(wandb_run, metric_name: str, edges: torch.Tensor, epoch: int):
    if not wandb_run or wandb is None:
        return
    try:
        wandb_run.log({f"probe/{metric_name}/bin_edges": edges.detach().cpu().numpy(), "epoch": epoch})
    except Exception:
        pass


# -------------------------
# Targets (tutte binnate)
# -------------------------
def _prepare_targets(feats: dict, mkey: str, n_bins: int):
    """
    Binna SEMPRE (anche 'labels') così tutte e tre le feature hanno lo stesso numero di livelli.
    Ritorna:
      y  (LongTensor, 0..n_bins-1),
      n_classes (int == n_bins),
      edges (Tensor n_bins+1),
      bin_names (list[str])  es. ['0.1-1.7', '1.7-3.2', ...]
    """
    vals = feats[mkey].to(torch.float32)
    y, edges = make_bin_labels(vals, n_bins=n_bins)
    bin_names = _format_bin_names(edges, precision=4)
    return y.long(), n_bins, edges, bin_names


# -------------------------
# Orchestratore
# -------------------------
def log_linear_probe(
    model,
    epoch: int,
    n_bins: int = 5,
    test_size: float = 0.2,
    steps: int = 1000,
    lr: float = 1e-2,
    rng_seed: int = 42,
    patience: int = 20,
    min_delta: float = 0.0,
    save_csv: bool = True,
    upto_layer: int | None = None,
    layer_tag: str | None = None,
):
    """
    Linear probe su: 'cum_area', 'convex_hull', 'labels' — tutte binnate in n_bins.
    - Usa TUTTI i campioni con split STRATIFICATO per bin/classe (default 80/20).
    - Early stopping su validation loss (NO logging per-step).
    - Logga su W&B:
        * SOLO accuracy finale
        * Confusion matrix come tabella (Table) leggibile
        * (opzionale) edges dei bin
    - Salva la confusion matrix anche in CSV (stile Excel) in model.arch_dir.
    """
    E, feats = compute_val_embeddings_and_features(model, upto_layer=upto_layer)   # E: [N, D]
    E_np = E.numpy()

    probe_targets = ["cum_area", "convex_hull", "labels"]
    if "density" in feats:
        probe_targets.append("density")

    summary_rows = []

    for mkey in probe_targets:
        # 1) target binned (stesso numero di livelli) + nomi dei bin
        y, n_classes, edges, bin_names = _prepare_targets(feats, mkey, n_bins=n_bins)
        metric_name = f"{layer_tag}/{mkey}" if layer_tag else mkey

        # 2) stratified split
        train_idx, test_idx = stratified_split(y, test_size=test_size, rng_seed=rng_seed)
        if len(train_idx) == 0 or len(test_idx) == 0:
            _log_accuracy_wandb(model.wandb_run, f"{metric_name}/warn_empty_split", 0.0, epoch)
            continue

        # 3) train & eval (full-batch + early stopping)
        Xtr, ytr = E_np[train_idx], y.numpy()[train_idx]
        Xte, yte = E_np[test_idx],  y.numpy()[test_idx]

        acc, y_true, y_pred = train_linear_classifier(
            Xtr, ytr, Xte, yte,
            device=model.device,
            n_classes=n_classes,
            max_steps=steps,
            lr=lr,
            weight_decay=0.0,
            patience=patience,
            min_delta=min_delta,
        )

        summary_rows.append((metric_name, acc))

        # 4) Confusion matrix "Excel-like" (DataFrame con nomi bin)
        df = _confusion_df(y_true, y_pred, n_classes, bin_names)

        # 5) Log: SOLO accuracy finale + tabella confusion + (edges opzionali) + CSV locale
        _log_accuracy_wandb(model.wandb_run, metric_name, acc, epoch)
        _log_confusion_table_wandb(model.wandb_run, df, metric_name, epoch)
        _log_bin_edges_wandb(model.wandb_run, metric_name, edges, epoch)

        if save_csv:
            csv_metric_name = metric_name.replace('/', '_')
            csv_path = _save_confusion_csv(df, model, csv_metric_name, epoch)
            # (facoltativo) potresti anche loggare il path su W&B:
            if model.wandb_run and wandb is not None:
                model.wandb_run.log({f"probe/{metric_name}/confusion_csv_path": csv_path, "epoch": epoch})

    if summary_rows and model.wandb_run and wandb is not None:
        labels = [name for name, _ in summary_rows]
        values = [val for _, val in summary_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        ax.bar(range(len(labels)), values, color='steelblue')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Linear probe summary @ epoch {epoch}")
        fig.tight_layout()
        model.wandb_run.log({f"probe/{layer_tag or 'top'}/summary": wandb.Image(fig)})
        plt.close(fig)


def log_joint_linear_probe(
    model,
    epoch: int,
    n_bins: int = 5,
    test_size: float = 0.2,
    steps: int = 1000,
    lr: float = 1e-2,
    rng_seed: int = 42,
    patience: int = 20,
    min_delta: float = 0.0,
    save_csv: bool = False,
    metric_prefix: str = "joint",
):
    """Linear probe for multimodal joint embeddings (image & text)."""
    E, feats = compute_joint_embeddings_and_features(model)
    if E.numel() == 0:
        return
    E_np = E.numpy()

    probe_targets = ["cum_area", "convex_hull", "labels"]
    if "density" in feats:
        probe_targets.append("density")

    summary_rows = []

    for mkey in probe_targets:
        y, n_classes, edges, bin_names = _prepare_targets(feats, mkey, n_bins=n_bins)
        metric_name = f"{metric_prefix}/{mkey}" if metric_prefix else mkey

        train_idx, test_idx = stratified_split(y, test_size=test_size, rng_seed=rng_seed)
        if len(train_idx) == 0 or len(test_idx) == 0:
            _log_accuracy_wandb(getattr(model, 'wandb_run', None), f"{metric_name}/warn_empty_split", 0.0, epoch)
            continue

        Xtr, ytr = E_np[train_idx], y.numpy()[train_idx]
        Xte, yte = E_np[test_idx],  y.numpy()[test_idx]

        acc, y_true, y_pred = train_linear_classifier(
            Xtr, ytr, Xte, yte,
            device=model.device,
            n_classes=n_classes,
            max_steps=steps,
            lr=lr,
            weight_decay=0.0,
            patience=patience,
            min_delta=min_delta,
        )

        summary_rows.append((metric_name, acc))

        df = _confusion_df(y_true, y_pred, n_classes, bin_names)

        wandb_run = getattr(model, 'wandb_run', None)
        _log_accuracy_wandb(wandb_run, metric_name, acc, epoch)
        _log_confusion_table_wandb(wandb_run, df, metric_name, epoch)
        _log_bin_edges_wandb(wandb_run, metric_name, edges, epoch)

        if save_csv:
            csv_metric_name = metric_name.replace('/', '_')
            csv_path = _save_confusion_csv(df, model, csv_metric_name, epoch)
            if wandb_run and wandb is not None:
                wandb_run.log({f"probe/{metric_name}/confusion_csv_path": csv_path, "epoch": epoch})

    if summary_rows and wandb_run and wandb is not None:
        labels = [name for name, _ in summary_rows]
        values = [val for _, val in summary_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        ax.bar(range(len(labels)), values, color='indianred')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Joint probe summary @ epoch {epoch}")
        fig.tight_layout()
        wandb_run.log({f"probe/{metric_prefix or 'joint'}/summary": wandb.Image(fig)})
        plt.close(fig)
