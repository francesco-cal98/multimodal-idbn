# trainer_multimodal.py
from pathlib import Path
import argparse, sys, yaml, torch
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / 'src'
for p in (PROJECT_ROOT, SRC_ROOT, SRC_ROOT / 'classes', SRC_ROOT / 'configs'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from src.classes.gdbn_model import iMDBN
from src.datasets.uniform_dataset import create_dataloaders_uniform

DEFAULT_CONFIG_PATH = SRC_ROOT / "configs" / "multimodal_training_config.yaml"


def parse_args():
    ap = argparse.ArgumentParser("Train a simple multimodal iDBN (image) + joint RBM with label softmax")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_params(cfg: dict) -> dict:
    t = cfg.get("training", {})
    return {
        # shared (immagine)
        "LEARNING_RATE": t.get("learning_rate", 0.1),
        "WEIGHT_PENALTY": t.get("weight_penalty", 1e-4),
        "INIT_MOMENTUM": t.get("init_momentum", 0.5),
        "FINAL_MOMENTUM": t.get("final_momentum", 0.95),
        "LEARNING_RATE_DYNAMIC": t.get("learning_rate_dynamic", True),
        "CD": t.get("cd", 1),

        # epochs
        "EPOCHS_IMG": t.get("epochs_image", 100),
        "EPOCHS_JOINT": t.get("epochs_joint", 200),

        # joint
        "JOINT_LEARNING_RATE": t.get("joint_learning_rate", t.get("learning_rate", 0.1)),
        "JOINT_CD": t.get("joint_cd", t.get("cd", 1)),
        "CROSS_GIBBS_STEPS": t.get("cross_gibbs_steps", 50),

        # aux clamped-CD
        "USE_AUX": t.get("use_aux", True),
        "JOINT_AUX_COND_STEPS": t.get("JOINT_AUX_COND_STEPS", t.get("aux_cond_steps", 50)),
        "JOINT_AUX_EVERY_K": t.get("JOINT_AUX_EVERY_K", 10),
        # logging
        "LOG_EVERY": t.get("log_every", 5),
        "LOG_EVERY_PCA": t.get("log_every_pca", 25),
        "LOG_EVERY_PROBE": t.get("log_every_probe", 10),
    }


def maybe_wandb(cfg: dict, params: dict):
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enable", False): 
        return None
    run = wandb.init(
        project=wcfg.get("project", "groundeep-diagnostics-multimodal"),
        entity=wcfg.get("entity"),
        name=wcfg.get("run_name"),
        config=cfg,            # << logga l’intera YAML
    )
    # aggiungi anche i parametri “build_params”
    run.config.update({"_derived_params": params}, allow_val_change=True)
    return run



def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset = cfg.get("dataset", {})
    model = cfg.get("model", {})
    params = build_params(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = create_dataloaders_uniform(
        data_path=dataset.get("path"),
        data_name=dataset.get("name"),
        batch_size=dataset.get("batch_size", 128),
        num_workers=dataset.get("num_workers", 1),
        multimodal_flag=dataset.get("multimodal_flag", True),
    )

    wandb_run = maybe_wandb(cfg,params)
    if wandb_run:
    # salva la config usata come file nel run
        wandb.save(str(args.config))  # la vedi nei "Files"

    # Istanza modello multimodale
    imdbn = iMDBN(
        layer_sizes_img=model.get("image_layers", [10000, 1500, 1500]),
        joint_layer_size=model.get("joint_hidden", 1000),
        params=params,
        dataloader=train_loader,
        val_loader=val_loader,
        device=device,
        num_labels=model.get("num_labels", 32),
        wandb_run=wandb_run,
    )

    # 1) iDBN visiva: carica se disponibile, altrimenti allena
    image_pre = cfg.get("paths", {}).get("image_idbn_pretrained")
    if image_pre:
        ok = imdbn.load_pretrained_image_idbn(image_pre)
        if not ok:
            print("[main] fallback: training image iDBN da zero...")
            imdbn.image_idbn.train(params["EPOCHS_IMG"],
                                   log_every_pca=params["LOG_EVERY_PCA"],
                                   log_every_probe=params["LOG_EVERY_PROBE"])
    else:
        print("Training image iDBN...")
        imdbn.image_idbn.train(params["EPOCHS_IMG"],
                               log_every_pca=params["LOG_EVERY_PCA"],
                               log_every_probe=params["LOG_EVERY_PROBE"])

    # (opzionale) fine-tuning ultimo RBM immagine
    ft_epochs = int(cfg.get("paths", {}).get("image_idbn_finetune_last_epochs", 0))
    if ft_epochs > 0:
        imdbn.finetune_image_last_layer(epochs=ft_epochs, lr_scale=0.3)

    # 2) Joint (autorecon first + condizionali via aux) + logging completo
    print("Training joint RBM...")
    imdbn.train_joint(epochs=params["EPOCHS_JOINT"],
                      log_every=params["LOG_EVERY"],
                      log_every_pca=params["LOG_EVERY_PCA"],
                      log_every_probe=params["LOG_EVERY_PROBE"])

    # Save
    save_dir = Path(cfg.get("paths", {}).get("save_dir", "./networks")).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.get('paths', {}).get('save_name', 'imdbn_trained')}.pkl"
    imdbn.save_model(str(save_path))
    print(f"✅ Saved multimodal model to {save_path}")

    if wandb_run: wandb_run.finish()


if __name__ == "__main__":
    main()
