# train_bimodal.py
"""
Training script for bimodal DBN (Numerosity + MNIST-100).
"""

from pathlib import Path
import argparse
import sys
import yaml
import torch
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imdbn.models.imdbn_bimodal import iMDBN_BiModal
from imdbn.datasets.uniform_dataset import create_dataloaders_uniform

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "bimodal_training_config.yaml"


def parse_args():
    ap = argparse.ArgumentParser("Train bimodal DBN (Numerosity + MNIST-100)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_params(cfg: dict) -> dict:
    t = cfg.get("training", {})
    return {
        # shared (both modalities)
        "LEARNING_RATE": t.get("learning_rate", 0.1),
        "WEIGHT_PENALTY": t.get("weight_penalty", 1e-4),
        "INIT_MOMENTUM": t.get("init_momentum", 0.5),
        "FINAL_MOMENTUM": t.get("final_momentum", 0.95),
        "LEARNING_RATE_DYNAMIC": t.get("learning_rate_dynamic", True),
        "CD": t.get("cd", 1),

        # epochs
        "EPOCHS_MOD1": t.get("epochs_mod1", 100),
        "EPOCHS_MOD2": t.get("epochs_mod2", 100),
        "EPOCHS_JOINT": t.get("epochs_joint", 200),

        # joint
        "JOINT_LEARNING_RATE": t.get("joint_learning_rate", t.get("learning_rate", 0.1)),
        "JOINT_CD": t.get("joint_cd", t.get("cd", 1)),
        "CROSS_GIBBS_STEPS": t.get("cross_gibbs_steps", 50),

        # aux clamped-CD
        "USE_AUX": t.get("use_aux", True),
        "JOINT_AUX_COND_STEPS": t.get("JOINT_AUX_COND_STEPS", t.get("aux_cond_steps", 50)),
        "JOINT_AUX_EVERY_K": t.get("JOINT_AUX_EVERY_K", 10),
    }


def build_logging_params(cfg: dict) -> dict:
    """Extract logging parameters from config."""
    log_cfg = cfg.get("logging", {})
    return {
        # Frequencies
        "LOG_EVERY": log_cfg.get("log_every", 5),
        "LOG_EVERY_PCA": log_cfg.get("log_every_pca", 25),
        "LOG_EVERY_PROBE": log_cfg.get("log_every_probe", 10),
        "LOG_EVERY_ENERGY": log_cfg.get("log_every_energy", 50),
        "LOG_EVERY_TRAJECTORY": log_cfg.get("log_every_trajectory", 50),
        "LOG_EVERY_NEIGHBORS": log_cfg.get("log_every_neighbors", 50),
        "LOG_EVERY_CONVERGENCE": log_cfg.get("log_every_convergence", 25),
    }


def maybe_wandb(cfg: dict, params: dict):
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enable", False):
        return None
    run = wandb.init(
        project=wcfg.get("project", "groundeep-bimodal"),
        entity=wcfg.get("entity"),
        name=wcfg.get("run_name"),
        config=cfg,
    )
    # aggiungi anche i parametri "build_params"
    run.config.update({"_derived_params": params}, allow_val_change=True)
    return run


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset = cfg.get("dataset", {})
    model = cfg.get("model", {})
    params = build_params(cfg)
    log_params = build_logging_params(cfg)
    logging_cfg = cfg.get("logging", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders with MNIST-100 support
    train_loader, val_loader, _ = create_dataloaders_uniform(
        data_path=dataset.get("path"),
        data_name=dataset.get("name"),
        batch_size=dataset.get("batch_size", 128),
        num_workers=dataset.get("num_workers", 1),
        multimodal_flag=dataset.get("multimodal_flag", True),
        second_modality=dataset.get("second_modality", "mnist100"),
        mnist100_path=dataset.get("mnist100_path"),
    )

    wandb_run = maybe_wandb(cfg, params)
    if wandb_run:
        # salva la config usata come file nel run
        wandb.save(str(args.config))

    # Istanza modello bimodale
    bimodal = iMDBN_BiModal(
        layer_sizes_mod1=model.get("mod1_layers", [10000, 1500, 500]),
        layer_sizes_mod2=model.get("mod2_layers", [1568, 500, 500]),
        joint_layer_size=model.get("joint_hidden", 500),
        params=params,
        dataloader=train_loader,
        val_loader=val_loader,
        device=device,
        wandb_run=wandb_run,
        logging_cfg=logging_cfg,
    )

    # 1) DBN Modality 1 (Numerosità): carica se disponibile, altrimenti allena
    mod1_pre = cfg.get("paths", {}).get("mod1_pretrained")
    if mod1_pre:
        ok = bimodal.load_pretrained_mod1_dbn(mod1_pre)
        if not ok:
            print("[main] fallback: training MOD1 iDBN da zero...")
            _train_mod1_dbn(bimodal, train_loader, params, device)
    else:
        print("Training MOD1 iDBN (Numerosità)...")
        _train_mod1_dbn(bimodal, train_loader, params, device)

    # (opzionale) fine-tuning ultimo RBM mod1
    ft_epochs_mod1 = int(cfg.get("paths", {}).get("mod1_finetune_last_epochs", 0))
    if ft_epochs_mod1 > 0:
        bimodal.finetune_mod1_last_layer(epochs=ft_epochs_mod1, lr_scale=0.3)

    # 2) DBN Modality 2 (MNIST-100): carica se disponibile, altrimenti allena
    mod2_pre = cfg.get("paths", {}).get("mod2_pretrained")
    if mod2_pre:
        ok = bimodal.load_pretrained_mod2_dbn(mod2_pre)
        if not ok:
            print("[main] fallback: training MOD2 iDBN da zero...")
            _train_mod2_dbn(bimodal, train_loader, params, device)
    else:
        print("Training MOD2 iDBN (MNIST-100)...")
        _train_mod2_dbn(bimodal, train_loader, params, device)

    # (opzionale) fine-tuning ultimo RBM mod2
    ft_epochs_mod2 = int(cfg.get("paths", {}).get("mod2_finetune_last_epochs", 0))
    if ft_epochs_mod2 > 0:
        bimodal.finetune_mod2_last_layer(epochs=ft_epochs_mod2, lr_scale=0.3)

    # 3) Joint training
    print("Training joint RBM...")
    bimodal.train_joint(
        epochs=params["EPOCHS_JOINT"],
        log_every=log_params["LOG_EVERY"],
        log_every_pca=log_params["LOG_EVERY_PCA"],
        log_every_probe=log_params["LOG_EVERY_PROBE"],
        log_every_trajectory=log_params["LOG_EVERY_TRAJECTORY"],
    )

    # Save
    save_dir = Path(cfg.get("paths", {}).get("save_dir", "./networks")).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.get('paths', {}).get('save_name', 'bimodal_trained')}.pkl"
    bimodal.save_model(str(save_path))
    print(f"Saved bimodal model to {save_path}")

    if wandb_run:
        wandb_run.finish()


def _create_modality_dataloader(train_loader, modality_idx, device):
    """
    Create a dataloader wrapper that yields only one modality.

    Args:
        train_loader: Original bimodal dataloader yielding (mod1, mod2)
        modality_idx: 0 for mod1, 1 for mod2
        device: torch device
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Collect all data for this modality
    all_data = []
    all_labels = []  # We need labels for iDBN training

    base_dataset = train_loader.dataset
    if hasattr(base_dataset, 'dataset'):
        # It's a Subset
        base = base_dataset.dataset
        indices = base_dataset.indices
    else:
        base = base_dataset
        indices = range(len(base))

    for idx in indices:
        sample = base[idx]
        all_data.append(sample[modality_idx])
        # Get the label for this sample
        label = base.labels[idx] if hasattr(base, 'labels') else idx
        all_labels.append(label)

    data_tensor = torch.stack(all_data)
    labels_tensor = torch.tensor(all_labels)

    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


def _train_mod1_dbn(bimodal, train_loader, params, device):
    """Train Modality 1 DBN on numerosity images using iDBN.train()."""

    # Create dedicated dataloader for mod1
    mod1_loader = _create_modality_dataloader(train_loader, 0, device)

    # Set the dataloader on the iDBN
    bimodal.mod1_dbn.dataloader = mod1_loader

    # Use the iDBN's native training method
    epochs = params["EPOCHS_MOD1"]
    print(f"  Training MOD1 iDBN for {epochs} epochs...")
    bimodal.mod1_dbn.train(epochs)


def _train_mod2_dbn(bimodal, train_loader, params, device):
    """Train Modality 2 DBN on MNIST-100 images using iDBN.train()."""

    # Create dedicated dataloader for mod2
    mod2_loader = _create_modality_dataloader(train_loader, 1, device)

    # Set the dataloader on the iDBN
    bimodal.mod2_dbn.dataloader = mod2_loader

    # Use the iDBN's native training method
    epochs = params["EPOCHS_MOD2"]
    print(f"  Training MOD2 iDBN for {epochs} epochs...")
    bimodal.mod2_dbn.train(epochs)


if __name__ == "__main__":
    main()
