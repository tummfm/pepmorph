#!/usr/bin/env python

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from common import DEFAULT_CVAE_CHECKPOINT, FIGS_DIR, GEN_PEPTIDES_DIR, TEAL, set_paper_style
from cvae.models import CVAESimpleEnc
from cvae.utils import ALPHABET, CONDITION_LENGTH, MAX_SEQ_LENGTH

FEATURES = ["length", "is_assembled", "ap", "has_beta_sheet_content", "hydrophobic_moment", "net_charge"]
MAX_FASTA_LENGTH = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent visualization for generated peptides.")
    parser.add_argument("--gen-dir", type=str, default=str(GEN_PEPTIDES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(FIGS_DIR))
    parser.add_argument(
        "--checkpoint-cvae",
        type=str,
        default=str(DEFAULT_CVAE_CHECKPOINT),
    )
    parser.add_argument("--plot-centers", action="store_true")
    return parser.parse_args()


def build_cond_mask_from_row(row: pd.Series, feature_order=FEATURES):
    cond = np.zeros(len(feature_order), dtype=np.float32)
    mask = np.zeros(len(feature_order), dtype=np.float32)
    for i, feat in enumerate(feature_order):
        if feat in row.index and pd.notna(row[feat]):
            val = float(row[feat])
            cond[i] = val / MAX_FASTA_LENGTH if feat == "length" else val
            mask[i] = 1.0
    return cond, mask


def project_2d(x, method="umap", random_state=42, n_neighbors=30, min_dist=0.15, perplexity=30):
    if x is None or len(x) == 0:
        return None, "None"
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="cosine",
                random_state=random_state,
            )
            y = reducer.fit_transform(x)
            return y, "UMAP"
        except Exception as exc:
            warnings.warn(f"UMAP unavailable ({exc}); falling back to t-SNE.")
    from sklearn.manifold import TSNE

    y = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=random_state,
    ).fit_transform(x)
    return y, "t-SNE"


def encode_with_cvae_encoder(cvae_model, seq_list, batch_size=256, device=None):
    m = cvae_model.module if hasattr(cvae_model, "module") else cvae_model
    m.eval()
    device = device or next(m.parameters()).device
    mus = []
    alphabet = ALPHABET
    batch_converter = alphabet.get_batch_converter()

    with torch.no_grad():
        for i in range(0, len(seq_list), batch_size):
            chunk = [(str(j), s) for j, s in enumerate(seq_list[i : i + batch_size])]
            _, _, toks = batch_converter(chunk)
            toks = toks[:, 1:-1]
            toks = toks.to(device)
            mu_enc, _ = m.encode(toks)
            mus.append(mu_enc.detach().cpu().numpy())
    return np.vstack(mus)


def label_success(target, is_target, rmoi):
    tgt = str(target).lower() if pd.notna(target) else None
    if pd.isna(rmoi):
        if tgt == "sphere":
            return "target_sphere"
        if tgt == "fiber":
            return "target_fiber"
        return "target_unknown"
    if tgt == "sphere":
        return "sphere_success" if is_target == "yes" else "sphere_fail"
    if tgt == "fiber":
        return "fiber_success" if is_target == "yes" else "fiber_fail"
    return "target_unknown"


def plot_scatter(df, xcol, ycol, title, filename):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    order = [
        "sphere_success",
        "fiber_success",
        "sphere_fail",
        "fiber_fail",
        "target_sphere",
        "target_fiber",
    ]
    palette = {
        "sphere_success": TEAL[1],
        "fiber_success": TEAL[5],
        "sphere_fail": TEAL[2],
        "fiber_fail": TEAL[4],
        "target_sphere": "#8aa3b0",
        "target_fiber": "#9aa1a6",
    }
    markers = {
        "sphere_success": "o",
        "fiber_success": "s",
        "sphere_fail": "^",
        "fiber_fail": "v",
        "target_sphere": ".",
        "target_fiber": "*",
    }
    sizes = {
        "sphere_success": 60,
        "fiber_success": 60,
        "sphere_fail": 52,
        "fiber_fail": 52,
        "target_sphere": 10,
        "target_fiber": 10,
    }
    for lab in order:
        sub = df[df["success_label"] == lab]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=sizes[lab],
            alpha=0.7,
            linewidths=0,
            color=palette[lab],
            marker=markers[lab],
            label=lab.replace("_", " "),
        )
    ax.set_xlabel(title)
    ax.set_ylabel("")
    sns.despine(ax=ax)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=10, loc="best")
    plt.tight_layout()
    fig.savefig(filename, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style()

    gen_dir = Path(args.gen_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rmoi_df = pd.read_csv(gen_dir / "rmoi_and_ap_by_run.csv")
    rmoi_df = rmoi_df.loc[:, ~rmoi_df.columns.str.contains("^Unnamed")]
    agg = rmoi_df.groupby("peptide").agg(
        mean_rmoi=("RMOI", "mean"),
        mean_ap=("aggregation_propensity", "mean"),
        is_target=("is_target", lambda x: x.mode()[0] if not x.mode().empty else False),
        morphology=("morphology", "first"),
    ).reset_index()
    agg = agg.rename(columns={"peptide": "sequence"})

    fibers = pd.read_csv(gen_dir / "fibers_with_conditions.csv")
    spheres = pd.read_csv(gen_dir / "spheres_with_conditions.csv")
    spheres_extra = gen_dir / "spheres_with_conditions_extra.csv"
    if spheres_extra.exists():
        spheres = pd.concat([spheres, pd.read_csv(spheres_extra)], ignore_index=True)
    spheres = spheres[spheres["length"] >= 5]

    for df in (fibers, spheres):
        if "peptide" in df.columns and "sequence" not in df.columns:
            df.rename(columns={"peptide": "sequence"}, inplace=True)

    fibers["target_morphology"] = "fiber"
    spheres["target_morphology"] = "sphere"

    sim_df = pd.concat([fibers, spheres], ignore_index=True)
    sim_df["sequence"] = sim_df["sequence"].astype(str).str.upper()

    sim_df = sim_df.merge(agg, on="sequence", how="left")
    sim_df["success_label"] = [
        label_success(t, v, r) for t, v, r in zip(sim_df["target_morphology"], sim_df["is_target"], sim_df["mean_rmoi"])
    ]

    for feat in FEATURES:
        if feat not in sim_df.columns:
            sim_df[feat] = np.nan

    conds, masks = zip(*[build_cond_mask_from_row(sim_df.loc[i, FEATURES], FEATURES) for i in range(len(sim_df))])
    cond_mat = torch.tensor(np.stack(conds, axis=0), dtype=torch.float32)
    mask_mat = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae_model = CVAESimpleEnc(
        encoder_hidden_dim=256,
        num_encoder_layers=2,
        vocab_size=len(ALPHABET),
        latent_dim=24,
        cond_dim=CONDITION_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        decoder_hidden_dim=256,
        num_decoder_layers=2,
        nhead=8,
        dropout=0.1,
    )
    state = torch.load(Path(args.checkpoint_cvae).expanduser(), map_location=device, weights_only=True)
    cvae_model.load_state_dict(state)
    cvae_model.eval()
    cvae_model.to(device)

    m = cvae_model.module if hasattr(cvae_model, "module") else cvae_model
    m.eval()
    with torch.no_grad():
        summary = m.compute_summary(cond_mat.to(device), mask_mat.to(device))
        prior_mu, _ = m.compute_prior(summary)
    prior_mu_np = prior_mu.detach().cpu().numpy()

    posterior_mu_np = encode_with_cvae_encoder(cvae_model, sim_df["sequence"].tolist(), batch_size=256, device=device)

    zp, meth_p = project_2d(prior_mu_np, method="umap")
    zq, meth_q = project_2d(posterior_mu_np, method="umap")

    sim_df["_Zp_x"] = zp[:, 0]
    sim_df["_Zp_y"] = zp[:, 1]
    sim_df["_Zq_x"] = zq[:, 0]
    sim_df["_Zq_y"] = zq[:, 1]

    plot_scatter(sim_df, "_Zp_x", "_Zp_y", f"{meth_p} of conditional prior mean", output_dir / "simulated_prior_umap.svg")
    plot_scatter(sim_df, "_Zq_x", "_Zq_y", f"{meth_q} of encoder posterior mean", output_dir / "simulated_posterior_umap.svg")

    if args.plot_centers:
        centers = prior_mu_np
        rs = np.random.default_rng(0)
        samples_per_cond = 40
        z_points = []
        z_colors = []
        targets = sim_df["target_morphology"].to_numpy()
        for u in range(len(centers)):
            mu = centers[u]
            eps = rs.standard_normal((samples_per_cond, mu.shape[0]))
            z = mu + eps
            z_points.append(z)
            z_colors.extend([targets[u]] * samples_per_cond)
        z_points = np.vstack(z_points)

        try:
            import umap

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine", random_state=42)
            centers_2d = reducer.fit_transform(centers)
            noise_2d = reducer.transform(z_points)
        except Exception:
            centers_2d, _ = project_2d(centers, method="tsne")
            noise_2d, _ = project_2d(z_points, method="tsne")

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        palette_cloud = {"sphere": TEAL[0], "fiber": TEAL[2]}
        for tgt in ["sphere", "fiber"]:
            mask = np.array(z_colors) == tgt
            if mask.sum():
                ax.scatter(
                    noise_2d[mask, 0],
                    noise_2d[mask, 1],
                    s=6,
                    alpha=0.25,
                    linewidths=0,
                    color=palette_cloud[tgt],
                    label=f"{tgt} prior samples",
                )

        for tgt, marker, col in [("sphere", "o", TEAL[3]), ("fiber", "s", TEAL[5])]:
            mask = targets == tgt
            if mask.sum():
                ax.scatter(
                    centers_2d[mask, 0],
                    centers_2d[mask, 1],
                    s=120,
                    alpha=0.95,
                    linewidth=0.5,
                    edgecolor="#333333",
                    color=col,
                    marker=marker,
                    label=f"{tgt} condition center",
                )

        sns.despine(ax=ax)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=10, loc="best")
        plt.tight_layout()
        fig.savefig(output_dir / "cond_prior_centers_plus_noise.svg", format="svg", bbox_inches="tight", dpi=300)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
