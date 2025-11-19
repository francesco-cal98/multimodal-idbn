import math
import wandb
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projection registration
import torch
import numpy as np
from scipy.stats import spearmanr

def log_reconstructions_to_wandb(original, reconstruction, step=0, num_images=8, name="reconstruction_grid"):
    """
    Log a grid of original and reconstructed images to Weights & Biases.
    """
    orig = torch.tensor((original[:num_images]).reshape(num_images,100,100)).unsqueeze(1)
    recon = torch.tensor((reconstruction[:num_images]).reshape(num_images,100,100)).unsqueeze(1)

    combined = torch.cat([val for pair in zip(orig, recon) for val in pair], dim=0)

    grid = vutils.make_grid(combined.unsqueeze(1), nrow=2, normalize=True)
    wandb.log({name: [wandb.Image(grid, caption=name)]})

def log_barplot(results, metric_name, arch_name, dist_name, ylabel="Value"):
    """
    results: dict con chiavi = bin string (es. "1-4") e valori = metric
    metric_name: nome della metrica (es. "linear_sep", "monotonicity", "silhouette")
    """
    bins = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(6,4))
    plt.bar(bins, values, color="steelblue", alpha=0.8)
    plt.ylabel(ylabel)
    plt.xlabel("Numerosity bins")
    plt.title(f"{metric_name} ({arch_name}, {dist_name})")
    plt.tight_layout()

    # log to wandb
    wandb.log({f"{arch_name}_{dist_name}_{metric_name}_bins_plot": wandb.Image(plt)})
    plt.close()


def plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name,wandb_run):
    """
    Genera plot 2D degli embedding colorati per feature e calcola le correlazioni.

    Args:
        emb_2d (np.array): Embedding ridotti a 2 dimensioni.
        features (dict): Dizionario di nomi di feature e array di valori.
        arch_name (str): Nome dell'architettura.
        dist_name (str): Nome della distribuzione.
        method_name (str): Nome del metodo di riduzione (e.g., 'PCA', 'UMAP').

    Returns:
        dict: Correlazioni Spearman tra dimensioni e feature.
    """
    print(f"  Generating 2D embedding plot for {arch_name}/{dist_name} using {method_name}...")
    
    if emb_2d.shape[0] == 0 or emb_2d.shape[1] != 2:
        print(f"    Skipping 2D embedding plot for {arch_name}/{dist_name}: Invalid 2D embeddings.")
        return {}

    correlations = {}
    n_features = len(features)
    
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    i = 0
    for feat_name, values in features.items():
        if i >= len(axs): 
            break
        
        if len(values) != emb_2d.shape[0] or len(values) < 2:
            print(f"    Feature '{feat_name}' length mismatch or insufficient data for embeddings. Skipping plot for this feature.")
            # Assicurati che i valori di correlazione siano NaN se non calcolabili
            correlations[f"{feat_name}_dim1"] = np.nan
            correlations[f"{feat_name}_dim2"] = np.nan
            i += 1 # Vai all'asse successivo
            continue

        rho_dim1, _ = spearmanr(emb_2d[:, 0], values)
        correlations[f"{feat_name}_dim1"] = rho_dim1
        rho_dim2, _ = spearmanr(emb_2d[:, 1], values)
        correlations[f"{feat_name}_dim2"] = rho_dim2

        if feat_name == "Labels":
            color_values = np.log(values)
        else:
            color_values = values

        sc = axs[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap='viridis', s=40, alpha=0.8)
        axs[i].set_title(f"Feature: {feat_name}\nDim1={correlations[f'{feat_name}_dim1']:.2f}, Dim2={correlations[f'{feat_name}_dim2']:.2f}")
        axs[i].set_xlabel(f"{method_name}-1")
        axs[i].set_ylabel(f"{method_name}-2")
        fig.colorbar(sc, ax=axs[i], label=feat_name)
        i += 1
    
    for j in range(i, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"{method_name} 2D Embedding for {arch_name} ({dist_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    wandb_run.log({f"embeddings/{dist_name}/{arch_name}/{method_name}_2d_embedding": wandb.Image(plt.gcf())})
    plt.close()
    return correlations


def plot_3d_embedding_and_correlations(emb_3d, features, arch_name, dist_name, method_name, wandb_run):
    """Generate 3D scatter plots for each feature and log Spearman correlations."""
    print(f"  Generating 3D embedding plot for {arch_name}/{dist_name} using {method_name}...")

    if emb_3d.shape[0] == 0 or emb_3d.shape[1] != 3:
        print(f"    Skipping 3D embedding plot for {arch_name}/{dist_name}: Invalid 3D embeddings.")
        return {}

    correlations = {}
    n_features = len(features)

    n_cols = 3
    n_rows = int(math.ceil(n_features / n_cols)) if n_features > 0 else 1

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for idx, (feat_name, values) in enumerate(features.items()):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        if len(values) != emb_3d.shape[0] or len(values) < 2:
            print(f"    Feature '{feat_name}' length mismatch or insufficient data for embeddings. Skipping plot for this feature.")
            correlations[f"{feat_name}_dim1"] = np.nan
            correlations[f"{feat_name}_dim2"] = np.nan
            correlations[f"{feat_name}_dim3"] = np.nan
            ax.axis('off')
            continue

        rho_dim1, _ = spearmanr(emb_3d[:, 0], values)
        rho_dim2, _ = spearmanr(emb_3d[:, 1], values)
        rho_dim3, _ = spearmanr(emb_3d[:, 2], values)
        correlations[f"{feat_name}_dim1"] = rho_dim1
        correlations[f"{feat_name}_dim2"] = rho_dim2
        correlations[f"{feat_name}_dim3"] = rho_dim3

        if feat_name == "Labels":
            color_values = np.log(values)
        else:
            color_values = values

        sc = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=color_values, cmap='viridis', s=30, alpha=0.8)
        ax.set_title(
            f"{feat_name}\nDim1={rho_dim1:.2f}, Dim2={rho_dim2:.2f}, Dim3={rho_dim3:.2f}"
        )
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
        ax.set_zlabel(f"{method_name}-3")
        fig.colorbar(sc, ax=ax, shrink=0.6, aspect=12, pad=0.1, label=feat_name)

    # Hide unused subplots
    total_plots = n_rows * n_cols
    for idx in range(n_features, total_plots):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        ax.axis('off')

    plt.suptitle(f"{method_name} 3D Embedding for {arch_name} ({dist_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    wandb_run.log({f"embeddings/{dist_name}/{arch_name}/{method_name}_3d_embedding": wandb.Image(fig)})
    plt.close(fig)
    return correlations
