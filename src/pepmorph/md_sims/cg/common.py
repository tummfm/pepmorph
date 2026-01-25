from __future__ import annotations

import sys
from pathlib import Path

CG_DIR = Path(__file__).resolve().parent
MD_SIMS_DIR = CG_DIR.parent
REPO_ROOT = CG_DIR.parents[3]
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CG_ARTIFACTS_DIR = ARTIFACTS_DIR / "md_sims" / "cg"
ROBUSTNESS_ARTIFACTS_DIR = CG_ARTIFACTS_DIR / "robustness"
PEPMORPH_MAIN_ARTIFACTS_DIR = CG_ARTIFACTS_DIR / "pepmorph-main"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pepmorph.shared.plot_style import TEAL, TEXT_COLOR, set_paper_style, teal_palette
