"""
Basic training example for Multimodal DBN (iMDBN).

This script demonstrates:
1. Loading data
2. Creating and training an iMDBN model
3. Cross-modal reconstruction
4. Saving and loading models
"""

import torch
import yaml
from pathlib import Path
from imdbn.models import iMDBN
from imdbn.datasets import create_dataloaders_uniform

def main():
    # Configuration
    config_path = Path("configs/multimodal_training_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders_uniform(
        path2data="path/to/your/data",  # Update this
        data_name="your_dataset_name",   # Update this
        batch_size=64,
        val_size=0.1
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")

    # Model hyperparameters
    params = {
        # Image DBN
        "LEARNING_RATE": config["training"]["learning_rate"],
        "WEIGHT_PENALTY": config["training"]["weight_penalty"],
        "INIT_MOMENTUM": config["training"]["init_momentum"],
        "FINAL_MOMENTUM": config["training"]["final_momentum"],
        "LEARNING_RATE_DYNAMIC": config["training"].get("learning_rate_dynamic", True),
        "CD": config["training"]["cd"],
        "EPOCHS_IMG": config["training"]["epochs_image"],

        # Joint RBM
        "EPOCHS_JOINT": config["training"]["epochs_joint"],
        "JOINT_LEARNING_RATE": config["training"]["joint_learning_rate"],
        "JOINT_CD": config["training"]["joint_cd"],
        "CROSS_GIBBS_STEPS": config["training"]["cross_gibbs_steps"],

        # Auxiliary training
        "USE_AUX": config["training"]["use_aux"],
        "JOINT_AUX_COND_STEPS": config["training"]["JOINT_AUX_COND_STEPS"],
        "JOINT_AUX_EVERY_K": config["training"].get("JOINT_AUX_EVERY_K", 1),
    }

    # Create model
    print("Creating iMDBN model...")
    model = iMDBN(
        layer_sizes_img=[10000, 1500, 500],  # Image DBN: input → 1500 → 500
        layer_sizes_txt_or_joint=256,         # Joint layer: 500+32 → 256
        params=params,
        dataloader=train_loader,
        val_loader=val_loader,
        device=device,
        num_labels=32,  # Number of classes
        wandb_run=None,  # Set to wandb.init() for logging
    )
    print(f"Architecture: {model.arch_str}")

    # Optional: Load pretrained image DBN
    # pretrained_path = "path/to/pretrained_image_dbn.pkl"
    # if Path(pretrained_path).exists():
    #     model.load_pretrained_image_idbn(pretrained_path)
    # else:
    #     # Train image DBN from scratch
    #     print("Training image DBN...")
    #     model.image_idbn.train(
    #         epochs=params["EPOCHS_IMG"],
    #         log_every_pca=25,
    #         log_every_probe=10
    #     )

    # For this example, we'll skip image training
    print("Skipping image DBN training (use pretrained or uncomment above)")

    # Initialize joint RBM biases
    print("Initializing joint RBM biases from data...")
    model.init_joint_bias_from_data(n_batches=10)

    # Train joint RBM
    print("Training joint RBM...")
    model.train_joint(
        epochs=params["EPOCHS_JOINT"],
        warmup_epochs=10,
        log_every_pca=10,
        log_every_metrics=5,
        log_every_cross=10,
        log_every_probe=10,
    )

    # Save model
    save_path = "imdbn_trained.pkl"
    print(f"Saving model to {save_path}...")
    model.save_model(save_path)

    # Demonstrate cross-modal reconstruction
    print("\nDemonstrating cross-modal reconstruction...")
    with torch.no_grad():
        # Get a test batch
        test_images, test_labels = next(iter(test_loader))
        test_images = test_images[:8].to(device)
        test_labels = test_labels[:8].to(device)

        # Extract image embeddings
        z_img = model.image_idbn.represent(test_images.view(test_images.size(0), -1))

        # Cross-modal reconstruction
        pred_labels, recon_z = model._cross_reconstruct(z_img, test_labels)

        # Image → Label accuracy
        pred_classes = pred_labels.argmax(dim=1)
        true_classes = test_labels.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean()
        print(f"IMG→TXT accuracy: {accuracy:.2%}")

        # Label → Image reconstruction
        recon_images = model.image_idbn.decode(recon_z)
        mse = torch.nn.functional.mse_loss(
            test_images.view(test_images.size(0), -1),
            recon_images
        )
        print(f"TXT→IMG MSE: {mse:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
