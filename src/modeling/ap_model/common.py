from __future__ import annotations

import sys
from pathlib import Path

AP_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = AP_MODEL_DIR.parents[2]
MODEL_DIR = AP_MODEL_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


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
