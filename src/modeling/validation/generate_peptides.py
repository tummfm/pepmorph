#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy

from common import DEFAULT_AP_CHECKPOINT, DEFAULT_CVAE_CHECKPOINT, GEN_PEPTIDES_DIR
from classifier.models import PeptidePredictor
from cvae.models import CVAESimpleEnc
from cvae.utils import (
    CONDITION_LENGTH,
    MAX_SEQ_LENGTH,
    PAD_TOKEN_ID,
    ALPHABET,
    BOS_ID,
    EOS_ID,
    idx_to_fasta,
    esm_model_pretrained,
    convert_and_pad,
)

FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

MAX_FASTA_LENGTH = 10

FIBER_RANGES = {
    "length": (7, 11, 1),
    "is_assembled": (1, 2, 1),
    "has_beta_sheet_content": (1, 2, 1),
    "net_charge": (0.4, 0.6, 0.05),
}

SPHERE_RANGES = {
    "length": (5, 8, 1),
    "is_assembled": (1, 2, 1),
    "hydrophobic_moment": (0.6, 1.05, 0.1),
    "net_charge": (0.4, 0.6, 0.05),
}

AMINO_ACID_DICT = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate peptides with the CVAE and filter with the AP predictor.")
    parser.add_argument("--mode", choices=["targeted", "unconditional", "all"], default="targeted")
    parser.add_argument("--output-dir", type=str, default=str(GEN_PEPTIDES_DIR))
    parser.add_argument(
        "--checkpoint-cvae",
        type=str,
        default=str(DEFAULT_CVAE_CHECKPOINT),
    )
    parser.add_argument(
        "--checkpoint-predictor",
        type=str,
        default=str(DEFAULT_AP_CHECKPOINT),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-per-cond-fiber", type=int, default=300)
    parser.add_argument("--n-per-cond-sphere", type=int, default=20)
    parser.add_argument("--ap-cutoff", type=float, default=0.43)
    parser.add_argument("--clas-cutoff", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--existing-spheres", type=str, default="")
    parser.add_argument("--existing-random-spheres", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_fasta(tokens: List[int]) -> str:
    fasta = ""
    for token in tokens:
        if token == EOS_ID:
            break
        if token in [0, 1, 3, 29, 30, 31, 32]:
            continue
        fasta += idx_to_fasta[token]
    return fasta


def generate_cond_mask_vectors(device, **kwargs):
    condition = torch.zeros(len(FEATURES), dtype=torch.float, device=device)
    mask = torch.zeros(len(FEATURES), dtype=torch.float, device=device)

    for idx, feature in enumerate(FEATURES):
        value = kwargs.get(feature, None)
        if value is not None:
            condition[idx] = value if feature != "length" else value / MAX_FASTA_LENGTH
            mask[idx] = 1.0
    return condition, mask


def sample_from_prior_ar(model, condition, mask, temperature: float = 1.0, n: int = 1):
    m = model.module if hasattr(model, "module") else model
    m.eval()

    if n == 1:
        condition = condition.unsqueeze(0)
        mask = mask.unsqueeze(0)
    else:
        condition = condition.repeat(n, 1)
        mask = mask.repeat(n, 1)

    device = next(m.parameters()).device
    condition, mask = condition.to(device), mask.to(device)
    summary = m.compute_summary(condition, mask)
    prior_mu, prior_logvar = m.compute_prior(summary)
    z = m.reparameterize(prior_mu, prior_logvar)

    seq = torch.full((condition.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(condition.size(0), dtype=torch.bool, device=device)
    outputs = torch.full((condition.size(0), MAX_SEQ_LENGTH), PAD_TOKEN_ID, dtype=torch.long, device=device)

    for t in range(MAX_SEQ_LENGTH):
        logits = m.decode(z, seq, summary)
        next_logits = logits[:, -1, :]
        if temperature == 1.0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        outputs[:, t] = next_token.squeeze(1)
        seq = torch.cat([seq, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == EOS_ID)
        if finished.all():
            break

    outs = []
    for row in outputs.tolist():
        if EOS_ID in row:
            cut = row.index(EOS_ID)
            toks = row[:cut]
        else:
            toks = row
        fasta = to_fasta(toks)
        outs.append((fasta, len(fasta)))
    return outs


def generate_samples_from_conditions(
    cvae_model,
    device,
    feature_ranges: Dict[str, List[float]],
    temperature: float = 1.0,
    n: int = 100,
    mask_out: bool = False,
) -> List[Tuple[str, int]]:
    results = []
    combos = list(product(*feature_ranges.values()))
    for combo in tqdm(combos, desc="Generating samples"):
        params = dict(zip(feature_ranges.keys(), combo))
        condition, mask = generate_cond_mask_vectors(device=device, **params)
        if mask_out:
            mask = torch.zeros_like(mask)
            mask[0] = 1.0
        samples = sample_from_prior_ar(cvae_model, condition, mask, temperature=temperature, n=n)
        results.extend(samples)
    return results


def convert_to_three_letter(seq: str) -> str:
    return "-".join([AMINO_ACID_DICT[aa] for aa in seq])


def classify_and_filter(
    model,
    batch_data: List[Tuple[str, int]],
    device: str,
    batch_size: int = 2048,
    ap_cutoff: float = 0.43,
    clas_cutoff: float = 0.75,
) -> List[Tuple[str, str, int, float, float]]:
    results = []
    model.to(device)
    model.eval()

    for start_idx in tqdm(range(0, len(batch_data), batch_size), desc="Classifying batches"):
        chunk = batch_data[start_idx : start_idx + batch_size]
        data = [(f"peptide_{start_idx + i}", seq) for i, (seq, _) in enumerate(chunk)]
        tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH).to(device)

        with torch.no_grad():
            ap_preds, cls_preds = model(tokens)

        for (seq, length), ap_pred, cls_pred in zip(chunk, ap_preds, cls_preds):
            ap_val = ap_pred.item()
            cls_val = cls_pred.item()
            if ap_val < ap_cutoff or cls_val < clas_cutoff:
                continue
            results.append((seq, convert_to_three_letter(seq), length, ap_val, cls_val))

    return results


def ranges_to_values(ranges: Dict[str, Tuple[float, float, float]]) -> Dict[str, List[float]]:
    return {key: np.arange(v[0], v[1], v[2]).tolist() for key, v in ranges.items()}


def write_sequence_file(path: Path, sequences: List[str]) -> None:
    with open(path, "w") as f:
        for seq in sequences:
            f.write(f"{seq}\n")


def write_filtered(path: Path, items: List[Tuple[str, str, int, float, float]]) -> None:
    with open(path, "w") as f:
        for seq, three_seq, _, ap, cls_ in items:
            f.write(f"{seq}, {three_seq}, {ap}, {cls_}\n")


def write_fasta(path: Path, sequences: List[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">peptide_{i}\n{seq}\n")


def run_targeted(
    cvae_model,
    predictor,
    device,
    output_dir: Path,
    temperature: float,
    n_fiber: int,
    n_sphere: int,
    ap_cutoff: float,
    clas_cutoff: float,
    existing_spheres: Path | None,
):
    fiber_vals = ranges_to_values(FIBER_RANGES)
    sphere_vals = ranges_to_values(SPHERE_RANGES)

    fiber_samples = generate_samples_from_conditions(
        cvae_model=cvae_model,
        device=device,
        feature_ranges=fiber_vals,
        temperature=temperature,
        n=n_fiber,
    )

    write_sequence_file(output_dir / "generated_fibers_init.txt", [seq for seq, _ in fiber_samples])

    fiber_valid = classify_and_filter(
        predictor, list(set(fiber_samples)), device=device, ap_cutoff=ap_cutoff, clas_cutoff=clas_cutoff
    )

    filtered_fiber = output_dir / "filtered_ap_peptides_fiber.txt"
    filtered_fiber_final = output_dir / "filtered_ap_peptides_fiber_final.txt"
    write_filtered(filtered_fiber, fiber_valid)
    write_filtered(filtered_fiber_final, fiber_valid)
    write_fasta(output_dir / "filtered_ap_peptides_fiber.fst", [item[0] for item in fiber_valid])
    write_fasta(output_dir / "filtered_ap_peptides_fiber_final.fst", [item[0] for item in fiber_valid])

    sphere_samples = generate_samples_from_conditions(
        cvae_model=cvae_model,
        device=device,
        feature_ranges=sphere_vals,
        temperature=temperature,
        n=n_sphere,
    )

    sphere_seqs = [seq for seq, _ in sphere_samples]
    if existing_spheres and existing_spheres.exists():
        existing = set(existing_spheres.read_text().splitlines())
        sphere_seqs = [s for s in sphere_seqs if s not in existing]
        write_sequence_file(output_dir / "generated_spheres_init_extra.txt", sphere_seqs)
    else:
        write_sequence_file(output_dir / "generated_spheres_init.txt", sphere_seqs)

    sphere_valid = classify_and_filter(
        predictor, list(set((seq, len(seq)) for seq in sphere_seqs)), device=device, ap_cutoff=ap_cutoff, clas_cutoff=clas_cutoff
    )

    filtered_sphere = output_dir / "filtered_ap_peptides_spheres.txt"
    filtered_sphere_final = output_dir / "filtered_ap_peptides_spheres_final.txt"
    write_filtered(filtered_sphere, sphere_valid)
    write_filtered(filtered_sphere_final, sphere_valid)
    write_fasta(output_dir / "filtered_ap_peptides_spheres.fst", [item[0] for item in sphere_valid])
    write_fasta(output_dir / "filtered_ap_peptides_spheres_final.fst", [item[0] for item in sphere_valid])


def run_unconditional(
    cvae_model,
    predictor,
    device,
    output_dir: Path,
    temperature: float,
    n_fiber: int,
    n_sphere: int,
    ap_cutoff: float,
    clas_cutoff: float,
    existing_spheres: Path | None,
):
    fiber_vals = ranges_to_values(FIBER_RANGES)
    sphere_vals = ranges_to_values(SPHERE_RANGES)

    random_fibers = generate_samples_from_conditions(
        cvae_model=cvae_model,
        device=device,
        feature_ranges=fiber_vals,
        temperature=temperature,
        n=n_fiber,
        mask_out=True,
    )

    write_sequence_file(output_dir / "generated_random_fibers_init.txt", [seq for seq, _ in random_fibers])

    random_fiber_valid = classify_and_filter(
        predictor, list(set(random_fibers)), device=device, ap_cutoff=ap_cutoff, clas_cutoff=clas_cutoff
    )

    write_filtered(output_dir / "filtered_ap_peptides_random_fiber.txt", random_fiber_valid)
    write_fasta(output_dir / "filtered_ap_peptides_random_fiber.fst", [item[0] for item in random_fiber_valid])

    random_spheres = generate_samples_from_conditions(
        cvae_model=cvae_model,
        device=device,
        feature_ranges=sphere_vals,
        temperature=temperature,
        n=n_sphere,
        mask_out=True,
    )

    sphere_seqs = [seq for seq, _ in random_spheres]
    if existing_spheres and existing_spheres.exists():
        existing = set(existing_spheres.read_text().splitlines())
        sphere_seqs = [s for s in sphere_seqs if s not in existing]
        write_sequence_file(output_dir / "generated_random_spheres_init_extra.txt", sphere_seqs)
    else:
        write_sequence_file(output_dir / "generated_random_spheres_init.txt", sphere_seqs)

    random_sphere_valid = classify_and_filter(
        predictor, list(set((seq, len(seq)) for seq in sphere_seqs)), device=device, ap_cutoff=ap_cutoff, clas_cutoff=clas_cutoff
    )

    write_filtered(output_dir / "filtered_ap_peptides_random_spheres_extra.txt", random_sphere_valid)
    write_fasta(output_dir / "filtered_ap_peptides_random_spheres_extra.fst", [item[0] for item in random_sphere_valid])


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

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
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        cvae_model = nn.DataParallel(cvae_model)
    cvae_model.to(device)

    predictor = PeptidePredictor(deepcopy(esm_model_pretrained), alphabet=ALPHABET)
    pred_state = torch.load(Path(args.checkpoint_predictor).expanduser(), map_location=device, weights_only=True)
    predictor.load_state_dict(pred_state)
    predictor.eval()
    predictor.to(device)

    existing_spheres = Path(args.existing_spheres).expanduser().resolve() if args.existing_spheres else None
    existing_random_spheres = (
        Path(args.existing_random_spheres).expanduser().resolve() if args.existing_random_spheres else None
    )

    if args.mode in {"targeted", "all"}:
        run_targeted(
            cvae_model,
            predictor,
            device,
            output_dir,
            args.temperature,
            args.n_per_cond_fiber,
            args.n_per_cond_sphere,
            args.ap_cutoff,
            args.clas_cutoff,
            existing_spheres,
        )

    if args.mode in {"unconditional", "all"}:
        run_unconditional(
            cvae_model,
            predictor,
            device,
            output_dir,
            args.temperature,
            args.n_per_cond_fiber,
            args.n_per_cond_sphere,
            args.ap_cutoff,
            args.clas_cutoff,
            existing_random_spheres,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
