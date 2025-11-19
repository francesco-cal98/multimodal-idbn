# src/utils/energy_utils.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    wandb = None


# ============================================================
# Free energy (RBM Bernoulli) e "class energies" F_k(z)
# ============================================================

@torch.no_grad()
def rbm_free_energy(rbm, v: torch.Tensor) -> torch.Tensor:
    """
    F(v) = - v^T b_v - sum_j softplus(b_h[j] + (v W)_j)
    v: [B, V] in [0,1] (può essere soft/mean-field)
    return: [B]
    """
    pre_h = F.linear(v, rbm.W.t(), rbm.hid_bias)       # v @ W + b_h
    term_hidden = F.softplus(pre_h).sum(dim=1)
    term_visible = (v * rbm.vis_bias).sum(dim=1)
    return -term_visible - term_hidden


@torch.no_grad()
def class_free_energies(joint_rbm, z_img_top: torch.Tensor, K: int, Dz: int) -> torch.Tensor:
    """
    F_k(z) = F([z, e_k]) per k=1..K — versione vettorizzata (senza costruire [B*K, ...]).
    z_img_top: [B, Dz]  ->  return: [B, K]
    """
    Wz = joint_rbm.W[:Dz, :]                 # [Dz, H]
    Wy = joint_rbm.W[Dz: Dz+K, :]            # [K,  H]
    bz = joint_rbm.vis_bias[:Dz]             # [Dz]
    by = joint_rbm.vis_bias[Dz: Dz+K]        # [K]
    bh = joint_rbm.hid_bias                  # [H]

    # termine visibile su z
    z_bz = (z_img_top * bz.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B,1]

    # preattivazioni nascoste: (z @ Wz) + b_h, poi shift di Wy per ogni classe
    pre_h_base = z_img_top @ Wz + bh.unsqueeze(0)   # [B, H]
    pre_h_all = pre_h_base.unsqueeze(1) + Wy.unsqueeze(0)  # [B, K, H]

    term_hidden = F.softplus(pre_h_all).sum(dim=2)  # [B, K]

    Fk = - (z_bz + by.unsqueeze(0)) - term_hidden   # [B, K]
    return Fk


# ============================================================
# Un passo "mean-field lite" su y (IMG→TXT)
# ============================================================

@torch.no_grad()
def _deterministic_img2txt_step(joint_rbm, v: torch.Tensor, Dz: int, K: int,
                                softmax_y: bool = True, sample_h: bool = False, sample_v: bool = False) -> torch.Tensor:
    """
    Step deterministico (mean-field "lite"):
      v -> h_prob -> v_prob, re-clamp z, rinormalizza y (softmax).
    v: [B, Dz+K]
    """
    # h|v
    h_prob = torch.sigmoid(F.linear(v, joint_rbm.W.t(), joint_rbm.hid_bias))
    h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
    # v|h
    v_prob = torch.sigmoid(F.linear(h, joint_rbm.W, joint_rbm.vis_bias))

    v_next = v_prob.clone()
    v_next[:, :Dz] = v[:, :Dz]                     # re-clamp z

    y = v_next[:, Dz:Dz+K]
    if softmax_y:
        y = torch.softmax(y, dim=1)
    else:
        y = y.clamp_(1e-6, 1 - 1e-6)
    if sample_v:
        # opzionale, ma di default evitiamo rumore
        idx = torch.distributions.Categorical(probs=y).sample()
        y = torch.zeros_like(y).scatter_(1, idx.view(-1, 1), 1.0)

    v_next[:, Dz:Dz+K] = y
    return v_next


# ============================================================
# Caso singolo (IMG→TXT) — versione minimale e veloce
# ============================================================

@torch.no_grad()
def trace_single_img2txt(model,
                         img: torch.Tensor,
                         lbl_onehot: torch.Tensor | None,
                         steps: int = 30,
                         eps_l1: float = 1e-3,
                         stable_steps: int = 3,
                         gap_thresh: float = 0.25):
    """
    Scopo: misurare quanti step servono perché y (distribuzione sulle classi)
    si stabilizzi su una previsione sensata, dato z clampato.

    Misure principali:
      - p_top1(t), p_top2(t), gap(t)=p1-p2
      - ΔF_pred(t) = F_{argmax(y_t)}(z) - min_k F_k(z)    (energia label scelta)
      - steps_to_converge basato su: L1 piccolo, pred stabile, gap alto
      - “facilità” intrinseca del caso: margin_energy = F(2)-F(1)
      - conf energetica finale: softmax(-F_k(z)) → fe_top1_final, fe_gap_final
    """
    device = model.device
    joint = model.joint_rbm

    # z dal ramo immagine
    x = img.view(img.size(0), -1).float().to(device) if img.dim() > 2 else img.float().to(device)
    z = model.image_idbn.represent(x).clamp(1e-6, 1 - 1e-6)  # [1, Dz]

    Dz = getattr(model, "Dz_img", z.size(1))
    K  = getattr(model, "num_labels", (lbl_onehot.size(1) if lbl_onehot is not None else 32))

    # 1) energie di classe UNA VOLTA (z clampato → F_k(z) costante durante la traccia)
    Fk = class_free_energies(joint, z, K, Dz).squeeze(0)   # [K]
    Fmin, kstar = torch.min(Fk, dim=0)
    top2 = torch.topk(Fk, k=2, largest=False).values
    margin_energy = float((top2[1] - top2[0]).item())      # F(2)-F(1)

    # 2) catena mean-field solo su y
    y = torch.full((1, K), 1.0 / K, device=device)
    v = torch.cat([z, y], dim=1)

    p_top1, p_top2, p_gap, p_gt = [], [], [], []
    deltaF_pred_traj = []

    y_prev = y.clone()
    pred_cur = int(y.argmax(dim=1).item())
    same_pred_streak = 0
    steps_to_conv = steps + 1  # "non convergente" di default

    gt = int(lbl_onehot.argmax(dim=1).item()) if lbl_onehot is not None else None

    for t in range(1, steps + 1):
        v = _deterministic_img2txt_step(joint, v, Dz, K, softmax_y=True, sample_h=False, sample_v=False)
        y = v[:, Dz:Dz+K]

        vals, _ = y.topk(2, dim=1)
        p1, p2 = float(vals[0, 0].item()), float(vals[0, 1].item())
        gap = p1 - p2
        p_top1.append(p1); p_top2.append(p2); p_gap.append(gap)
        if gt is not None:
            p_gt.append(float(y[0, gt].item()))

        pred_new = int(y.argmax(dim=1).item())
        same_pred_streak = same_pred_streak + 1 if pred_new == pred_cur else 1
        pred_cur = pred_new

        # ΔF_pred(t) con Fk(z) precomputata
        deltaF_pred_traj.append(float((Fk[pred_cur] - Fmin).item()))

        # criterio di convergenza (semplice e robusto)
        l1 = float((y - y_prev).abs().sum().item())
        if (l1 < eps_l1) and (same_pred_streak >= stable_steps) and (pred_cur == int(kstar.item()) or gap >= gap_thresh):
            steps_to_conv = t
            break
        y_prev = y.clone()

    # finali
    predT = pred_cur
    fe_probs = torch.softmax(-Fk, dim=0)
    fe_top1_final = float(fe_probs.max().item())
    fe_gap_final  = float((fe_probs.topk(2).values[0] - fe_probs.topk(2).values[1]).item())

    return {
        # dinamica/confidenza
        "deltaF_pred_traj": deltaF_pred_traj,
        "deltaF_pred_final": deltaF_pred_traj[-1] if len(deltaF_pred_traj) else None,
        "p_top1": p_top1,
        "p_top2": p_top2,
        "p_gap":  p_gap,
        "p_gt":   p_gt if gt is not None else None,
        # finali/conf
        "p_top1_final": p_top1[-1] if len(p_top1) else float(1.0/K),
        "p_gap_final":  p_gap[-1]  if len(p_gap)  else 0.0,
        "fe_top1_final": fe_top1_final,
        "fe_gap_final":  fe_gap_final,
        # convergenza + “facilità”
        "steps_to_converge": steps_to_conv,
        "kstar": int(kstar.item()),
        "predT": predT,
        "margin_energy": margin_energy,
        # label GT
        "gt": gt,
    }


# ============================================================
# Scelta di un sample di validazione FISSO (coerente across epoche)
# ============================================================

@torch.no_grad()
def pick_fixed_val_case(model, target_label: int | None = None, within_batch_index: int = 0):
    """
    Ritorna (img[1,...], lbl_onehot[1,K]) dal validation set e lo *cacha* in model._fixed_val_case
    così da riutilizzare sempre lo stesso sample ad ogni epoca.
    """
    device = model.device
    if hasattr(model, "_fixed_val_case") and model._fixed_val_case is not None:
        img_cpu, lbl_cpu = model._fixed_val_case
        return img_cpu.to(device), lbl_cpu.to(device)

    if model.val_loader is None:
        raise RuntimeError("model.val_loader is None")

    chosen_img, chosen_lbl = None, None
    if target_label is None:
        for imgs, lbls in model.val_loader:
            chosen_img = imgs[within_batch_index:within_batch_index+1].cpu()
            chosen_lbl = lbls[within_batch_index:within_batch_index+1].cpu()
            break
    else:
        for imgs, lbls in model.val_loader:
            idx = (lbls.argmax(dim=1) == target_label).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                i0 = int(idx[0].item())
                chosen_img = imgs[i0:i0+1].cpu()
                chosen_lbl = lbls[i0:i0+1].cpu()
                break
        if chosen_img is None:  # fallback
            imgs, lbls = next(iter(model.val_loader))
            chosen_img = imgs[:1].cpu()
            chosen_lbl = lbls[:1].cpu()

    model._fixed_val_case = (chosen_img, chosen_lbl)
    return chosen_img.to(device), chosen_lbl.to(device)


# Retrocompatibilità con vecchio import: pick_val_case(...)
@torch.no_grad()
def pick_val_case(model, target_label: int | None = None, batch_idx: int = 0, within_batch_index: int = 0):
    """
    Alias retro-compatibile: ignora batch_idx e usa il caching del caso fisso.
    """
    return pick_fixed_val_case(model, target_label=target_label, within_batch_index=within_batch_index)


# ============================================================
# Logging W&B per il caso fisso
# ============================================================

@torch.no_grad()
def log_single_case_energy(model, case_dict: dict, epoch: int, tag: str = "fixed_case"):
    """
    Log minimale:
      - ΔF_pred(t) vs step
      - p_top1/p_top2/(p_gt) vs step
      - Summary con conf finale, steps_to_converge, kstar/predT, margin_energy
    """
    if getattr(model, "wandb_run", None) is None or wandb is None:
        return

    # 1) ΔF_pred(t)
    if "deltaF_pred_traj" in case_dict and len(case_dict["deltaF_pred_traj"]) > 0:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        xs = range(1, 1 + len(case_dict["deltaF_pred_traj"]))
        ax1.plot(xs, case_dict["deltaF_pred_traj"])
        ax1.set_xlabel("step")
        ax1.set_ylabel("ΔF_pred = F_yhat - F_min")
        ax1.set_title("Energia label (IMG→TXT)")
        model.wandb_run.log({f"case/{tag}/deltaF_pred_vs_steps": wandb.Image(fig1), "epoch": epoch})
        plt.close(fig1)

    # 2) p_top1 / p_top2 / p(y_true)
    if "p_top1" in case_dict and "p_top2" in case_dict:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(range(1, 1 + len(case_dict["p_top1"])), case_dict["p_top1"], label="p_top1")
        ax2.plot(range(1, 1 + len(case_dict["p_top2"])), case_dict["p_top2"], label="p_top2")
        if case_dict.get("p_gt", None) is not None and len(case_dict["p_gt"]) > 0:
            ax2.plot(range(1, 1 + len(case_dict["p_gt"])), case_dict["p_gt"], label="p(y_true)", linestyle="--")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("step")
        ax2.set_ylabel("probability")
        ax2.set_title("Confidenza nel tempo (IMG→TXT)")
        ax2.legend()
        model.wandb_run.log({f"case/{tag}/p_curves": wandb.Image(fig2), "epoch": epoch})
        plt.close(fig2)

    # 3) Summary
    txt = {
        "gt": case_dict.get("gt", None),
        "kstar": case_dict.get("kstar", None),
        "predT": case_dict.get("predT", None),
        "steps_to_converge": case_dict.get("steps_to_converge", None),

        "p_top1_final": case_dict.get("p_top1_final", None),
        "p_gap_final": case_dict.get("p_gap_final", None),
        "fe_top1_final": case_dict.get("fe_top1_final", None),
        "fe_gap_final": case_dict.get("fe_gap_final", None),

        "deltaF_pred_final": case_dict.get("deltaF_pred_final", None),
        "margin_energy": case_dict.get("margin_energy", None),
    }
    model.wandb_run.log({f"case/{tag}/summary": txt, "epoch": epoch})


# ============================================================
# Helper: run + log sul caso fisso (richiamalo ogni N epoche)
# ============================================================

@torch.no_grad()
def run_and_log_fixed_case(model, epoch: int,
                           target_label: int | None = None,
                           within_batch_index: int = 0,
                           steps: int = 30,
                           tag: str = "fixed"):
    """
    Esegue la traccia IMG→TXT sul caso fisso e logga i grafici/mining.
    Ritorna il dict con le metriche del caso.
    """
    img, lbl = pick_fixed_val_case(model, target_label=target_label, within_batch_index=within_batch_index)
    case = trace_single_img2txt(model, img, lbl, steps=steps)
    log_single_case_energy(model, case, epoch=epoch, tag=tag)
    return case
