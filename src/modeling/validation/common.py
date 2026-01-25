from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

VALIDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_DIR.parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
VALIDATION_ARTIFACTS = ARTIFACTS_DIR / "validation"
MODEL_DIR = VALIDATION_DIR.parent
SRC_DIR = REPO_ROOT / "src"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
DEFAULT_CVAE_CHECKPOINT = MODEL_ARTIFACTS_DIR / "masked_cvae" / "finetuned_cvae.pt"
DEFAULT_AP_CHECKPOINT = MODEL_ARTIFACTS_DIR / "ap_model" / "peptide_predictor.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.plot_style import TEAL, set_paper_style, teal_palette


def _install_alias(alias: str, target_module: object, submodules: tuple[str, ...]) -> None:
    sys.modules[alias] = target_module
    for sub in submodules:
        mod = getattr(target_module, sub, None)
        if mod is not None:
            sys.modules[f"{alias}.{sub}"] = mod


try:
    import masked_cvae  # type: ignore
    _install_alias("cvae", masked_cvae, ("models", "utils", "datasets"))
except Exception:
    pass

try:
    import ap_model  # type: ignore
    _install_alias("classifier", ap_model, ("models", "datasets"))
except Exception:
    pass

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = VALIDATION_ARTIFACTS / "results"
FIGS_DIR = VALIDATION_ARTIFACTS / "figs"
GEN_PEPTIDES_DIR = VALIDATION_ARTIFACTS / "gen_peptides"

FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]


def resolve_path(path: Optional[str], default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser().resolve()


def load_split_indices(splits_dir: Optional[Path] = None) -> dict[str, list[int]] | None:
    splits_dir = splits_dir or SPLITS_DIR
    train_path = splits_dir / "train_idx.txt"
    val_path = splits_dir / "val_idx.txt"
    test_path = splits_dir / "test_idx.txt"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        return None

    def _read(path: Path) -> list[int]:
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]

    return {"train": _read(train_path), "val": _read(val_path), "test": _read(test_path)}
