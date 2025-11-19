# src/utils/cross_steps_utils.py
# Misura "quanti step servono" a convergere in conditional Gibbs, senza usare la free energy.

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    wandb = None


@torch.no_grad()
def _gibbs_conditional_step(rbm, v, v_known, known_mask, sample_h=False, sample_v=False):
    """
    Un singolo step di Gibbs condizionale, allineato al corpo del tuo `conditional_gibbs`:
      - usa rbm.forward per p(h|v)
      - usa rbm.visible_probs (rispetta softmax_groups) per p(v|h)
      - opzionalmente campiona con rbm.sample_visible
      - re-clampa ai valori ORIGINALI v_known (non a v del passo precedente)
    """
    # h | v
    h_prob = rbm.forward(v)  # = sigmoid(v @ W + b_h)
    h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob

    # v | h
    v_prob = rbm.visible_probs(h)  # rispetta i blocchi softmax del label
    v_next = rbm.sample_visible(v_prob) if sample_v else v_prob

    # re-clamp ai noti (z o y) dai valori ORIGINALI
    v_next = v_next * (1 - known_mask) + v_known * known_mask
    return v_next, v_prob


# ------------------------------
# IMG -> TXT (z clampato)
# ------------------------------
def trace_img2txt_cross(
    model,
    img,
    lbl_onehot=None,
    max_steps=70,
    sample_h=False,
    sample_v=False,
    eps_l1=1e-3,
    stable_steps=3,
    gap_thresh=0.25,
):
    dev = model.device
    # --- prepara z dal ramo immagine ---
    x = img.view(img.size(0), -1).float().to(dev) if img.dim() > 2 else img.float().to(dev)
    z = model.image_idbn.represent(x)  # [1, Dz]

    Dz = getattr(model, "Dz_img", z.size(1))
    K  = (lbl_onehot.size(1) if lbl_onehot is not None
          else getattr(model, "num_labels", 32))
    V  = Dz + K

    # --- stato noto/maschera: clamp su z ---
    v_known = torch.zeros(1, V, device=dev)
    v_known[:, :Dz] = z
    known_mask = torch.zeros_like(v_known); known_mask[:, :Dz] = 1.0

    # --- stato iniziale: noti = v_known, ignoti random ---
    v = v_known * known_mask + (1 - known_mask) * torch.rand_like(v_known)

    # mezzo step "pulito" solo per inizializzare la baseline di misura
    h0 = model.joint_rbm.forward(v)
    v_prob0 = model.joint_rbm.visible_probs(h0)
    y_prev = v_prob0[:, Dz:]

    pred_cur = int(y_prev.argmax(dim=1).item())
    same_pred_streak = 0
    steps_to_conv = max_steps + 1

    p_top1, p_top2, p_gap, p_gt = [], [], [], []
    l1_list = []
    top1_idx_seq, top2_idx_seq = [], []
    gt_idx = int(lbl_onehot.argmax(dim=1).item()) if lbl_onehot is not None else None

    for t in range(1, max_steps + 1):
        v, v_prob = _gibbs_conditional_step(
            model.joint_rbm, v, v_known, known_mask,
            sample_h=sample_h, sample_v=sample_v
        )

        y_soft = v_prob[:, Dz:]
        vals, idxs = y_soft.topk(2, dim=1)
        p1, p2 = float(vals[0, 0].item()), float(vals[0, 1].item())
        k1, k2 = int(idxs[0, 0].item()), int(idxs[0, 1].item())
        gap = p1 - p2

        p_top1.append(p1); p_top2.append(p2); p_gap.append(gap)
        top1_idx_seq.append(k1); top2_idx_seq.append(k2)
        if gt_idx is not None:
            p_gt.append(float(y_soft[0, gt_idx].item()))

        l1 = float((y_soft - y_prev).abs().sum().item())
        l1_list.append(l1)

        pred_new = int(y_soft.argmax(dim=1).item())
        same_pred_streak = same_pred_streak + 1 if pred_new == pred_cur else 1
        pred_cur = pred_new

        if (l1 < eps_l1) and (same_pred_streak >= stable_steps) and (gap >= gap_thresh):
            steps_to_conv = t
            break

        y_prev = y_soft.clone()

    return {
        "dir": "img2txt",
        "steps_to_converge": steps_to_conv,
        "p_top1": p_top1,
        "p_top2": p_top2,
        "p_gap":  p_gap,
        "p_gt":   p_gt if (lbl_onehot is not None) else None,
        "l1":     l1_list,
        "predT":  pred_cur,
        # NUOVO: indici di classe per le curve
        "top1_idx": top1_idx_seq,
        "top2_idx": top2_idx_seq,
        "gt_idx": gt_idx,
    }


# ------------------------------
# TXT -> IMG (y clampato)
# ------------------------------
@torch.no_grad()
def trace_txt2img_cross(
    model,
    img,                    # [1, C, H, W] o [1, D] (per MSE verso GT)
    lbl_onehot,             # [1, K]
    max_steps=70,
    sample_h=False,
    sample_v=False,
    eps_z=1e-3,
    mse_tol=1e-5,
    patience=3,
    ema_beta: float = 0.0,  # opzionale: smoothing su z per stabilizzare la misura
):
    """
    Conditional Gibbs con y clampato (TXT->IMG), senza free energy:
    - misura Δz_l2 = ||z_t - z_{t-1}||_2 e MSE dell'immagine ricostruita vs GT
    - convergenza quando: Δz_l2 < eps_z e MSE non migliora più (miglioramento < mse_tol) per 'patience' step.
    - usa uno step condizionale allineato al tuo RBM (visible_probs + re-clamp su v_known)
    """
    dev = model.device

    # GT image flatten per MSE
    img_gt = img.to(dev).view(img.size(0), -1).float()

    # dimensioni
    Dz = getattr(model, "Dz_img", int(model.image_idbn.layers[-1].num_hidden))
    K  = getattr(model, "num_labels", lbl_onehot.size(1))
    V  = Dz + K

    # stato visibile noto/ignoto
    v_known = torch.zeros(1, V, device=dev)
    v_known[:, Dz:] = lbl_onehot.to(dev).float()         # clamp y
    known_mask = torch.zeros_like(v_known); known_mask[:, Dz:] = 1.0

    # stato iniziale: noti = v_known, ignoti random
    v = v_known.clone()
    try:
        # indice classe (se multi-hot/soft, prendi argmax)
        y_idx = int(lbl_onehot.argmax(dim=1).item())
        if hasattr(model, "z_class_mean"):
            v[:, :Dz] = model.z_class_mean[y_idx].unsqueeze(0)  # prior di classe su z
        else:
            # fallback "pulito" (mean-field di base)
            h0 = model.joint_rbm.forward(v_known)
            v_prob0 = model.joint_rbm.visible_probs(h0)
            v = v_prob0 * (1 - known_mask) + v_known * known_mask
    except Exception:
        h0 = model.joint_rbm.forward(v_known)
        v_prob0 = model.joint_rbm.visible_probs(h0)
        v = v_prob0 * (1 - known_mask) + v_known * known_mask

    # tracker
    # partiamo dal 'soft' per il confronto L2; prima di iterare lo inizializziamo leggendo p(v|h)
    z_prev = v[:, :Dz].clone()
    z_l2_list, img_mse_list = [], []
    best_mse = float("inf"); no_improve = 0
    steps_to_conv = max_steps + 1

    for t in range(1, max_steps + 1):
        # singolo step condizionale coerente col tuo RBM
        v, v_prob = _gibbs_conditional_step(
            model.joint_rbm, v, v_known, known_mask,
            sample_h=sample_h, sample_v=sample_v
        )

        # usiamo la stima "pulita" per verificare convergenza
        z_soft = v_prob[:, :Dz]

        # opzionale smoothing su z per stabilizzare la dinamica durante la misura
        if ema_beta > 0.0:
            z_new = (1.0 - ema_beta) * z_prev + ema_beta * z_soft
        else:
            z_new = z_soft

        # decodifica e MSE
        img_rec = model.image_idbn.decode(z_new).view_as(img_gt)
        mse = F.mse_loss(img_rec, img_gt).item()
        img_mse_list.append(mse)

        # variazione sul top-code
        dz = float(torch.norm((z_new - z_prev), p=2).item())
        z_l2_list.append(dz)
        z_prev = z_new.detach()

        # early stop: variazione z bassa e MSE che non migliora
        if dz < eps_z:
            if mse + 1e-12 < best_mse - mse_tol:
                best_mse = mse
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                steps_to_conv = t
                break
        else:
            # se z si muove ancora, aggiorna il best_mse e resetta
            if mse + 1e-12 < best_mse - mse_tol:
                best_mse = mse
            no_improve = 0

    return {
        "dir": "txt2img",
        "steps_to_converge": steps_to_conv,
        "z_l2": z_l2_list,
        "image_mse": img_mse_list,
        "best_mse": best_mse,
    }

# ------------------------------
# Sample fisso (cache) + logging W&B
# ------------------------------
@torch.no_grad()
def pick_fixed_val_case(model, target_label: int | None = None, within_batch_index: int = 0):
    """Sceglie e cacha un sample di validazione, così resta lo stesso tra le epoche."""
    dev = model.device
    if hasattr(model, "_fixed_val_case") and model._fixed_val_case is not None:
        img_cpu, lbl_cpu = model._fixed_val_case
        return img_cpu.to(dev), lbl_cpu.to(dev)

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
        if chosen_img is None:
            imgs, lbls = next(iter(model.val_loader))
            chosen_img = imgs[:1].cpu()
            chosen_lbl = lbls[:1].cpu()

    model._fixed_val_case = (chosen_img, chosen_lbl)
    return chosen_img.to(dev), chosen_lbl.to(dev)


@torch.no_grad()
def log_cross_case(model, out_img2txt: dict, out_txt2img: dict, epoch: int, tag: str):
    if getattr(model, "wandb_run", None) is None or wandb is None:
        return

    class_names = getattr(model, "class_names", None)  # opzionale

    # IMG -> TXT
    if out_img2txt and len(out_img2txt.get("p_top1", [])) > 0:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        xs = range(1, 1 + len(out_img2txt["p_top1"]))
        ax1.plot(xs, out_img2txt["p_top1"], label="p_top1")
        ax1.plot(xs, out_img2txt["p_top2"], label="p_top2")
        if out_img2txt.get("p_gt", None):
            ax1.plot(xs, out_img2txt["p_gt"], label="p(y_true)", linestyle="--")
        # annota le classi finali nel titolo
        k1_final = out_img2txt["top1_idx"][-1]
        k2_final = out_img2txt["top2_idx"][-1]
        def to_name(k): 
            if class_names and 0 <= k < len(class_names): return f"{k}:{class_names[k]}"
            return str(k)
        ax1.set_ylim(0, 1); ax1.set_xlabel("step"); ax1.set_ylabel("prob")
        ax1.set_title(f"IMG→TXT (Gibbs) — final top1={to_name(k1_final)}, top2={to_name(k2_final)}")
        ax1.legend()
        model.wandb_run.log({f"cross/{tag}/img2txt_p": wandb.Image(fig1), "epoch": epoch})
        plt.close(fig1)

        # gap
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(xs, out_img2txt["p_gap"], label="gap=p1-p2")
        ax2.set_xlabel("step"); ax2.set_ylabel("gap"); ax2.set_title("IMG→TXT gap")
        model.wandb_run.log({f"cross/{tag}/img2txt_gap": wandb.Image(fig2), "epoch": epoch})
        plt.close(fig2)

        # NUOVO: tabella step→classi e probabilità
        try:
            cols = ["step", "top1_idx", "p_top1", "top2_idx", "p_top2"]
            if out_img2txt.get("p_gt", None) is not None:
                cols += ["y_true_idx", "p_y_true"]
            if class_names:
                cols += ["top1_label", "top2_label"]
                if out_img2txt.get("p_gt", None) is not None:
                    cols += ["y_true_label"]

            tbl = wandb.Table(columns=cols)
            T = len(out_img2txt["p_top1"])
            gt_idx = out_img2txt.get("gt_idx", None)
            for t in range(T):
                r = [
                    t+1,
                    out_img2txt["top1_idx"][t], out_img2txt["p_top1"][t],
                    out_img2txt["top2_idx"][t], out_img2txt["p_top2"][t],
                ]
                if out_img2txt.get("p_gt", None) is not None:
                    r += [gt_idx, out_img2txt["p_gt"][t]]
                if class_names:
                    r += [
                        class_names[out_img2txt["top1_idx"][t]],
                        class_names[out_img2txt["top2_idx"][t]],
                    ]
                    if out_img2txt.get("p_gt", None) is not None and gt_idx is not None:
                        r += [class_names[gt_idx]]
                tbl.add_data(*r)
            model.wandb_run.log({f"cross/{tag}/img2txt_topk_table": tbl, "epoch": epoch})
        except Exception:
            pass

    # TXT -> IMG (come prima)
    if out_txt2img:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        xs = range(1, 1 + len(out_txt2img["image_mse"]))
        ax3.plot(xs, out_txt2img["image_mse"])
        ax3.set_xlabel("step"); ax3.set_ylabel("MSE"); ax3.set_title("TXT→IMG (Gibbs) MSE vs GT")
        model.wandb_run.log({f"cross/{tag}/txt2img_mse": wandb.Image(fig3), "epoch": epoch})
        plt.close(fig3)

    # Summary come prima (invariato)
    summary = {
        "img2txt_steps": out_img2txt.get("steps_to_converge", None) if out_img2txt else None,
        "txt2img_steps": out_txt2img.get("steps_to_converge", None) if out_txt2img else None,
        "txt2img_best_mse": out_txt2img.get("best_mse", None) if out_txt2img else None,
        "img2txt_pred_final": out_img2txt.get("predT", None) if out_img2txt else None,
        "img2txt_gt": out_img2txt.get("gt_idx", None) if out_img2txt else None,
    }
    model.wandb_run.log({f"cross/{tag}/summary": summary, "epoch": epoch})


@torch.no_grad()
def run_and_log_cross_fixed_case(
    model,
    epoch: int,
    target_label: int | None = None,
    within_batch_index: int = 0,
    max_steps: int = 70,
    sample_h: bool = False,
    sample_v: bool = False,
    tag: str = "fixed_cross",
):
    """Esegue entrambi i versi (IMG→TXT, TXT→IMG) sullo stesso sample fisso e logga i risultati."""
    img, lbl = pick_fixed_val_case(model, target_label=target_label, within_batch_index=within_batch_index)

    out_img2txt = trace_img2txt_cross(
        model, img, lbl_onehot=lbl, max_steps=max_steps,
        sample_h=sample_h, sample_v=sample_v
    )
    out_txt2img = trace_txt2img_cross(
        model, img, lbl_onehot=lbl, max_steps=max_steps,
        sample_h=sample_h, sample_v=sample_v
    )
    log_cross_case(model, out_img2txt, out_txt2img, epoch=epoch, tag=tag)
    return out_img2txt, out_txt2img
    # === APPENDI IN FONDO A src/utils/cross_steps_utils.py =========================
import numpy as np

@torch.no_grad()
def build_or_get_fixed_val_panel(model, per_class: int = 4):
    """
    Costruisce (e cacha) un pannello fisso di validazione con 'per_class' sample per classe.
    Ritorna tensori stacked (imgs, lbls). Se il val set è sbilanciato, usa ciò che trova.
    """
    dev = model.device
    if hasattr(model, "_fixed_val_panel") and model._fixed_val_panel is not None:
        imgs_cpu, lbls_cpu = model._fixed_val_panel
        return imgs_cpu.to(dev), lbls_cpu.to(dev)

    if model.val_loader is None:
        raise RuntimeError("val_loader is None")

    K = getattr(model, "num_labels", 32)
    buckets = [[] for _ in range(K)]

    for imgs, lbls in model.val_loader:
        B = imgs.size(0)
        for i in range(B):
            cls = int(lbls[i].argmax(dim=0).item())
            if len(buckets[cls]) < per_class:
                buckets[cls].append( (imgs[i:i+1].cpu(), lbls[i:i+1].cpu()) )
        if all(len(b)==per_class or len(b)>0 for b in buckets) and sum(len(b) for b in buckets) >= K:
            # abbiamo almeno un esempio per ogni classe presente + limiti raggiunti
            if all(len(b) >= per_class for b in buckets):
                break

    # fallback: concatena ciò che hai trovato (può mancare qualche classe)
    imgs_list, lbls_list = [], []
    for b in buckets:
        imgs_list.extend([x for (x,_) in b])
        lbls_list.extend([y for (_,y) in b])
    if len(imgs_list) == 0:
        # ultima risorsa: prendi il primo batch
        imgs, lbls = next(iter(model.val_loader))
        imgs_list = [imgs[:1].cpu()]
        lbls_list = [lbls[:1].cpu()]

    imgs_cpu = torch.cat(imgs_list, dim=0)
    lbls_cpu = torch.cat(lbls_list, dim=0)
    model._fixed_val_panel = (imgs_cpu, lbls_cpu)
    return imgs_cpu.to(dev), lbls_cpu.to(dev)


@torch.no_grad()
def _steps_stats(steps_list, max_steps):
    """Statistiche su steps_to_converge (solo campioni converged)."""
    arr = np.asarray(steps_list, dtype=np.int32)
    conv_mask = arr <= max_steps
    conv_vals = arr[conv_mask]
    stats = {
        "n_total": int(arr.size),
        "n_converged": int(conv_vals.size),
        "frac_converged": float(conv_vals.size / max(1, arr.size)),
        "mean": float(conv_vals.mean()) if conv_vals.size else None,
        "p50": float(np.percentile(conv_vals, 50)) if conv_vals.size else None,
        "p95": float(np.percentile(conv_vals, 95)) if conv_vals.size else None,
    }
    return stats, conv_mask


def _plot_steps_hist_with_nc(steps_list, max_steps, title):
    """Istogramma 1..max_steps + bin finale 'NC'."""
    arr = np.asarray(steps_list, dtype=np.int32)
    counts = []
    labels = []
    for s in range(1, max_steps + 1):
        counts.append(int((arr == s).sum()))
        labels.append(str(s))
    counts.append(int((arr > max_steps).sum()))
    labels.append("NC")

    fig, ax = plt.subplots(figsize=(min(12, max_steps * 0.35 + 2), 3.2))
    ax.bar(np.arange(len(labels)), counts)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Gibbs steps")
    ax.set_ylabel("# samples")
    ax.set_title(title)
    return fig


@torch.no_grad()
def run_and_log_cross_panel(
    model,
    epoch: int,
    per_class: int = 4,          # es. 4 per classe -> fino a 128 sample
    max_steps: int = 70,
    sample_h: bool = False,
    sample_v: bool = False,
    tag: str = "panel",
):
    """
    Esegue IMG→TXT e TXT→IMG su un pannello fisso (per_class per classe),
    aggrega e logga: istogrammi steps, summary (mean/p50/p95/frac_converged),
    e medie di p_top1/p_gap (IMG→TXT) + best MSE (TXT→IMG).
    """
    dev = model.device
    imgs, lbls = build_or_get_fixed_val_panel(model, per_class=per_class)
    N = imgs.size(0)

    i2t_steps, t2i_steps = [], []
    i2t_p1_final, i2t_gap_final = [], []
    t2i_best_mse = []

    for i in range(N):
        img1 = imgs[i:i+1].to(dev)
        lbl1 = lbls[i:i+1].to(dev)

        out_i2t = trace_img2txt_cross(
            model, img1, lbl_onehot=lbl1, max_steps=max_steps,
            sample_h=sample_h, sample_v=sample_v
        )
        out_t2i = trace_txt2img_cross(
            model, img1, lbl_onehot=lbl1, max_steps=max_steps,
            sample_h=sample_h, sample_v=sample_v
        )

        i2t_steps.append(int(out_i2t["steps_to_converge"]))
        t2i_steps.append(int(out_t2i["steps_to_converge"]))

        if len(out_i2t.get("p_top1", [])) > 0:
            i2t_p1_final.append(float(out_i2t["p_top1"][-1]))
        if len(out_i2t.get("p_gap", [])) > 0:
            i2t_gap_final.append(float(out_i2t["p_gap"][-1]))
        t2i_best_mse.append(float(out_t2i.get("best_mse", float("inf"))))

    # Statistiche globali
    i2t_stats, _ = _steps_stats(i2t_steps, max_steps)
    t2i_stats, _ = _steps_stats(t2i_steps, max_steps)
    mean_p1 = float(np.mean(i2t_p1_final)) if i2t_p1_final else None
    mean_gap = float(np.mean(i2t_gap_final)) if i2t_gap_final else None
    mean_best_mse = float(np.mean(t2i_best_mse)) if t2i_best_mse else None

    # Log su W&B
    if getattr(model, "wandb_run", None) is not None and wandb is not None:
        fig_i2t = _plot_steps_hist_with_nc(i2t_steps, max_steps, "IMG→TXT panel: steps to converge")
        model.wandb_run.log({f"conv/panel/{tag}/img2txt_steps_hist": wandb.Image(fig_i2t), "epoch": epoch})
        plt.close(fig_i2t)

        fig_t2i = _plot_steps_hist_with_nc(t2i_steps, max_steps, "TXT→IMG panel: steps to converge")
        model.wandb_run.log({f"conv/panel/{tag}/txt2img_steps_hist": wandb.Image(fig_t2i), "epoch": epoch})
        plt.close(fig_t2i)

        summary = {
            "img2txt/mean": i2t_stats["mean"],
            "img2txt/p50":  i2t_stats["p50"],
            "img2txt/p95":  i2t_stats["p95"],
            "img2txt/frac_converged": i2t_stats["frac_converged"],
            "txt2img/mean": t2i_stats["mean"],
            "txt2img/p50":  t2i_stats["p50"],
            "txt2img/p95":  t2i_stats["p95"],
            "txt2img/frac_converged": t2i_stats["frac_converged"],
            "img2txt/p_top1_final_mean": mean_p1,
            "img2txt/p_gap_final_mean":  mean_gap,
            "txt2img/best_mse_mean":     mean_best_mse,
            "n_total": i2t_stats["n_total"],
        }
        model.wandb_run.log({f"conv/panel/{tag}/summary": summary, "epoch": epoch})

    return {
        "img2txt": {"steps": i2t_steps, "stats": i2t_stats, "p1_mean": mean_p1, "gap_mean": mean_gap},
        "txt2img": {"steps": t2i_steps, "stats": t2i_stats, "best_mse_mean": mean_best_mse},
    }
# ===============================================================================
def run_and_log_z_mismatch_check(model, epoch: int, max_steps: int = 20, sample_h: bool = False, sample_v: bool = False, tag: str = "z_check"):
    """
    Confronta le distribuzioni del top-code z nei due casi:
      - z_img: estratto dalla iDBN dato l'input immagine
      - z_y:   ottenuto da TXT→IMG (clamp y, evolve z) fino a convergenza (mean-field di default)

    Logga su W&B: mean/std per componente (aggregati), cosine(z_y, z_img), e istogrammi di z.
    """
    if getattr(model, "wandb_run", None) is None:
        return

    dev = model.device
    try:
        imgs, lbls = next(iter(model.val_loader))
    except Exception:
        return
    imgs = imgs.to(dev)
    lbls = lbls.to(dev).float()
    B = imgs.size(0)

    # z da immagine
    z_img = model.image_idbn.represent(imgs.view(B, -1))  # [B, Dz]
    Dz = z_img.size(1)

    # z da testo: run TXT→IMG deterministica
    z_y_list = []
    for i in range(B):
        out = trace_txt2img_cross(
            model,
            img=imgs[i:i+1],
            lbl_onehot=lbls[i:i+1],
            max_steps=max_steps,
            sample_h=sample_h,
            sample_v=sample_v,
            eps_z=1e-3, mse_tol=1e-5, patience=3,
            ema_beta=0.0
        )
        # prendi l'ultimo z usato per il decode nell'ultima iterazione:
        # lo ricostruiamo replicando un passo con clamping e leggendo v_prob[:, :Dz]
        K = getattr(model, "num_labels", lbls.size(1))
        V = Dz + K
        v_known = torch.zeros(1, V, device=dev)
        v_known[:, Dz:] = lbls[i:i+1]
        known_mask = torch.zeros_like(v_known); known_mask[:, Dz:] = 1.0
        # stato finale: facciamo max_steps passi per riallineare v (deterministici, costo trascurabile sul singolo sample)
        v = v_known * known_mask + (1 - known_mask) * torch.rand_like(v_known)
        for _ in range(max_steps):
            v, v_prob = _gibbs_conditional_step(model.joint_rbm, v, v_known, known_mask, sample_h=sample_h, sample_v=sample_v)
        z_final = v_prob[:, :Dz]
        z_y_list.append(z_final)
    z_y = torch.cat(z_y_list, dim=0)  # [B, Dz]

    # statistiche globali
    def _stats(t):
        return {
            "mean": float(t.mean().item()),
            "std":  float(t.std(unbiased=False).item()),
            "q10":  float(t.quantile(0.10).item()),
            "q90":  float(t.quantile(0.90).item()),
        }
    stats_img = _stats(z_img)
    stats_zy  = _stats(z_y)

    # cosine per-sample tra z_y e z_img (quanto sono allineati)
    z_img_u = z_img / (z_img.norm(dim=1, p=2, keepdim=True) + 1e-12)
    z_y_u   = z_y   / (z_y.norm(dim=1, p=2, keepdim=True) + 1e-12)
    cosine  = (z_img_u * z_y_u).sum(dim=1).clamp(-1, 1)  # [B]
    cos_mean = float(cosine.mean().item())

    # Log W&B
    model.wandb_run.log({f"zcheck/{tag}/z_img_stats": stats_img, "epoch": epoch})
    model.wandb_run.log({f"zcheck/{tag}/z_y_stats":  stats_zy,  "epoch": epoch})
    model.wandb_run.log({f"zcheck/{tag}/cosine_mean": cos_mean, "epoch": epoch})

    try:
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.hist(z_img.flatten().cpu().numpy(), bins=50, alpha=0.6, label="z_img")
        ax1.hist(z_y.flatten().cpu().numpy(),  bins=50, alpha=0.6, label="z_y")
        ax1.set_title("Histogram z values"); ax1.legend()
        model.wandb_run.log({f"zcheck/{tag}/hist": wandb.Image(fig1), "epoch": epoch})
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.hist(cosine.cpu().numpy(), bins=30)
        ax2.set_title("cos(z_y, z_img) per sample")
        model.wandb_run.log({f"zcheck/{tag}/cosine_hist": wandb.Image(fig2), "epoch": epoch})
        plt.close(fig2)
    except Exception:
        pass
