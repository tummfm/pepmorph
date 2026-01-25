from __future__ import annotations

from typing import List, Optional


TEXT_COLOR = "#3F3F3F"
TEAL = ["#b5d1ae", "#80ae9a", "#568b87", "#326b77", "#1b485e", "#122740"]


def set_paper_style(use_seaborn: bool = True) -> None:
    from matplotlib import rcParams

    rcParams["font.family"] = "Lato"
    rcParams["font.size"] = 16
    rcParams["text.color"] = TEXT_COLOR
    rcParams["axes.labelcolor"] = TEXT_COLOR
    rcParams["xtick.color"] = TEXT_COLOR
    rcParams["ytick.color"] = TEXT_COLOR

    if not use_seaborn:
        return

    try:
        import seaborn as sns
    except Exception:
        return

    sns.set(style="ticks", context="talk")


def teal_palette(n: Optional[int] = None, include_light: bool = False) -> List[str]:
    base = ["#e2eddf"] + TEAL if include_light else list(TEAL)

    if n is None:
        return list(base)
    if n <= len(base):
        return list(base[:n])
    return list(base) + [base[-1]] * (n - len(base))
