#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import platform
import sys
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

CG_DIR = Path(__file__).resolve().parents[1]
if str(CG_DIR) not in sys.path:
    sys.path.insert(0, str(CG_DIR))

from common import PEPMORPH_MAIN_ARTIFACTS_DIR, TEAL, TEXT_COLOR, set_paper_style

DEFAULT_RUNS = ["run_1", "run_2", "run_3"]
DEFAULT_CLASSES = [
    "random",
    "unsupervised",
    "sphere_no_ap",
    "sphere_no_desc",
    "sphere_pepmorph",
    "fiber_no_ap",
    "fiber_no_desc",
    "fiber_pepmorph",
]

COHORT_LABELS = {
    "random": "Random",
    "unsupervised": "Unsupervised",
    "sphere_no_ap": "Failed AP screen",
    "sphere_no_desc": "Failed descriptor screen",
    "sphere_pepmorph": "PepMorph candidates",
    "fiber_no_ap": "Failed AP screen",
    "fiber_no_desc": "Failed descriptor screen",
    "fiber_pepmorph": "PepMorph candidates",
}

SECTION_LABELS = {
    "random": "Untargeted",
    "unsupervised": "Untargeted",
    "sphere_no_ap": "Spherical-targeted",
    "sphere_no_desc": "Spherical-targeted",
    "sphere_pepmorph": "Spherical-targeted",
    "fiber_no_ap": "Fibril-targeted",
    "fiber_no_desc": "Fibril-targeted",
    "fiber_pepmorph": "Fibril-targeted",
}

VIEW_1 = dict(degx=5, degy=100, degz=0, order="zyx", pad=1.1)
VIEW_2 = dict(degx=5, degy=100, degz=90, order="zyx", pad=1.1)

RENDER_DPI_SCALE = 0.55
WINDOW_SIZE_PX = (2200, 2200)
PAPER_GRAY = TEXT_COLOR
BB_COLOR = TEAL[4]
SC_COLOR = TEAL[3]

BB_POINT_SIZE = 8
SC_POINT_SIZE = 6

CUTOFF_NM = 1.8
MIN_PNG_BYTES = 2000

TOP_NAME = "peptide-cg.gro"
TRAJ_NAME = "trajout.xtc"

IMG_CACHE_DIRNAME = "_render_cache"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

warnings.filterwarnings(
    "ignore",
    message=r"Failed to use notebook backend:.*",
    category=UserWarning,
)


def is_valid_png(path: Path, min_bytes=MIN_PNG_BYTES):
    try:
        path = Path(path)
        if not path.exists():
            return False
        if path.stat().st_size < min_bytes:
            return False
        with path.open("rb") as handle:
            return handle.read(8) == _PNG_MAGIC
    except Exception:
        return False


def _residue_anchors(universe):
    """One anchor per residue (BB if present; else residue COM). Units: Angstrom."""
    anchors = []
    for res in universe.residues:
        bb = res.atoms.select_atoms("name BB")
        anchors.append(bb.positions[0] if bb.n_atoms else res.atoms.center_of_mass())
    return np.asarray(anchors, dtype=float)


def _largest_residue_cluster_from_anchors(anchors_a, cutoff_nm, box):
    """
    anchors_a: (n,3) Angstrom
    cutoff_nm: nm
    box: MDAnalysis ts.dimensions (Angstrom + angles)
    """
    cutoff_a = cutoff_nm * 10.0
    pairs = capped_distance(anchors_a, anchors_a, cutoff_a, box=box, return_distances=False)

    n = len(anchors_a)
    parent = np.arange(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        if i != j:
            union(int(i), int(j))

    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)

    largest = max(comps.values(), key=len) if comps else [0]
    return np.array(largest, dtype=int)


def center_main_cluster_inplace_on_current_frame(universe, cutoff_nm=CUTOFF_NM, use_bb_for_com=True):
    """
    Mutates positions of universe.atoms in the current timestep:
      1) make residues in largest cluster contiguous (min image)
      2) translate cluster COM to box center (min image)
      3) wrap into primary box
    """
    box = universe.dimensions.copy()
    anchors_all = _residue_anchors(universe)

    cluster_res_idx = _largest_residue_cluster_from_anchors(anchors_all, cutoff_nm, box)

    idx = np.concatenate([universe.residues[i].atoms.indices for i in cluster_res_idx])
    cluster = universe.atoms[idx]
    com_group = cluster.select_atoms("name BB") if use_bb_for_com else cluster

    com = com_group.center_of_mass()
    for ri in cluster_res_idx:
        res = universe.residues[int(ri)]
        disp = res.atoms.center_of_mass() - com
        shift = minimize_vectors(disp[None, :], box=box)[0] - disp
        res.atoms.positions += shift

    com2 = com_group.center_of_mass()
    box_center = 0.5 * box[:3]
    delta = minimize_vectors((com2 - box_center)[None, :], box=box)[0]
    universe.atoms.positions -= delta

    universe.atoms.pack_into_box(box=box)


def cell_vectors_from_dimensions(dim):
    """
    MDAnalysis dimensions: [a,b,c,alpha,beta,gamma]
      a,b,c in Angstrom; angles in degrees.
    Return 3 cell vectors (Angstrom) in Cartesian coordinates.
    """
    a, b, c, alpha, beta, gamma = dim
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=float)

    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-8)
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))
    vc = np.array([cx, cy, cz], dtype=float)

    return va, vb, vc


def unitcell_edges(va, vb, vc):
    """Edges for the parallelepiped cell."""
    o = np.array([0.0, 0.0, 0.0])
    pts = [
        o,
        va,
        vb,
        va + vb,
        vc,
        va + vc,
        vb + vc,
        va + vb + vc,
    ]
    edges_idx = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [(pts[i], pts[j]) for i, j in edges_idx]


def quat_axis_angle(axis, deg):
    ax, ay, az = axis
    n = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    th = math.radians(deg) * 0.5
    s = math.sin(th)
    return [ax * s, ay * s, az * s, math.cos(th)]


def quat_euler(degx=0.0, degy=0.0, degz=0.0, order="zyx"):
    def qmul(q2, q1):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        ]

    qx = quat_axis_angle([1, 0, 0], degx)
    qy = quat_axis_angle([0, 1, 0], degy)
    qz = quat_axis_angle([0, 0, 1], degz)
    q = [0, 0, 0, 1]
    for c in order.lower():
        q = qmul({"x": qx, "y": qy, "z": qz}[c], q)
    return q


def quat_to_rotmat(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def render_two_groups_pyvista(
    bb_pos,
    sc_pos,
    dims,
    out_png: Path,
    view_params,
    bb_color=BB_COLOR,
    sc_color=SC_COLOR,
    line_color=PAPER_GRAY,
    bb_point_size=BB_POINT_SIZE,
    sc_point_size=SC_POINT_SIZE,
    window_size_px=WINDOW_SIZE_PX,
    dpi_scale=RENDER_DPI_SCALE,
    off_screen=True,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    w, h = window_size_px
    w = int(w * dpi_scale)
    h = int(h * dpi_scale)

    pl = pv.Plotter(off_screen=off_screen, window_size=(w, h))
    pl.set_background("white")
    pl.disable_anti_aliasing()

    if bb_pos is not None and len(bb_pos) > 0:
        bb = pv.PolyData(bb_pos)
        pl.add_mesh(
            bb,
            color=bb_color,
            opacity=0.98,
            render_points_as_spheres=True,
            point_size=bb_point_size,
        )

    if sc_pos is not None and len(sc_pos) > 0:
        sc = pv.PolyData(sc_pos)
        pl.add_mesh(
            sc,
            color=sc_color,
            opacity=0.9,
            render_points_as_spheres=True,
            point_size=sc_point_size,
        )

    va, vb, vc = cell_vectors_from_dimensions(dims)
    for p0, p1 in unitcell_edges(va, vb, vc):
        pl.add_mesh(pv.Line(p0, p1), color=line_color, line_width=2)

    pl.camera.parallel_projection = True

    if "cam_dir" in view_params:
        cam_dir = np.asarray(view_params["cam_dir"], float)
        cam_up = np.asarray(view_params.get("cam_up", safe_up_for(cam_dir)), float)
    else:
        q = quat_euler(view_params["degx"], view_params["degy"], view_params["degz"], view_params["order"])
        rmat = quat_to_rotmat(q)
        cam_dir = rmat @ np.array([0.0, 0.0, 1.0])
        cam_up = rmat @ np.array([0.0, 1.0, 0.0])

    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)
    cam_right = np.cross(cam_dir, cam_up)
    cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-12)

    pts = []
    if bb_pos is not None and len(bb_pos) > 0:
        pts.append(bb_pos)
    if sc_pos is not None and len(sc_pos) > 0:
        pts.append(sc_pos)
    pts = np.vstack(pts) if len(pts) else np.zeros((1, 3), dtype=float)

    focal = pts.mean(axis=0)

    rel = pts - focal
    x = rel @ cam_right
    y = rel @ cam_up
    half_span = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))

    pad = float(view_params.get("pad", 1.12))
    pl.camera.parallel_scale = half_span * pad

    dist = 8.0 * half_span
    pl.camera.position = (focal + cam_dir * dist).tolist()
    pl.camera.focal_point = focal.tolist()
    pl.camera.up = cam_up.tolist()

    pl.show(screenshot=str(out_png), interactive=False, auto_close=True)

    if not is_valid_png(out_png):
        raise RuntimeError(f"Screenshot failed or invalid PNG: {out_png}")


def principal_axes(pts):
    """
    pts: (N,3) array in Angstrom.
    Returns (axes, evals) where axes[:,0] is the longest-axis direction.
    """
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0, keepdims=True)
    x = pts - c
    cov = (x.T @ x) / max(len(x), 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    evecs = evecs / (np.linalg.norm(evecs, axis=0, keepdims=True) + 1e-12)
    return evecs, evals


def safe_up_for(cam_dir, preferred=np.array([0.0, 1.0, 0.0])):
    cam_dir = np.asarray(cam_dir, float)
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    up = preferred.copy()
    if abs(np.dot(cam_dir, up)) > 0.92:
        up = np.array([1.0, 0.0, 0.0])
    up = up - np.dot(up, cam_dir) * cam_dir
    up = up / (np.linalg.norm(up) + 1e-12)
    return up


def render_two_views_for_peptide(
    peptide_dir: Path,
    out_png1: Path,
    out_png2: Path,
    cutoff_nm=CUTOFF_NM,
    top_name=TOP_NAME,
    traj_name=TRAJ_NAME,
    off_screen=True,
):
    top = peptide_dir / top_name
    traj = peptide_dir / traj_name
    if not top.exists() or not traj.exists():
        raise FileNotFoundError(f"Missing {top_name} or {traj_name} in {peptide_dir}")

    universe = mda.Universe(str(top), str(traj))
    universe.trajectory[-1]
    center_main_cluster_inplace_on_current_frame(universe, cutoff_nm=cutoff_nm)

    dims = universe.dimensions.copy()
    bb_pos = universe.atoms.select_atoms("name BB").positions.copy()
    sc_pos = universe.atoms.select_atoms("not name BB").positions.copy()

    pts = np.vstack([bb_pos, sc_pos]) if (len(bb_pos) and len(sc_pos)) else (bb_pos if len(bb_pos) else sc_pos)

    axes, evals = principal_axes(pts)
    a1 = axes[:, 0]
    a2 = axes[:, 1]
    a3 = axes[:, 2]

    anis = float(evals[0] / max(evals[1], 1e-12))

    if anis > 1.35:
        view1 = dict(cam_dir=a3, cam_up=safe_up_for(a3, preferred=a2), pad=VIEW_1.get("pad", 1.05))
        view2 = dict(cam_dir=a2, cam_up=safe_up_for(a2, preferred=a1), pad=VIEW_2.get("pad", 1.05))
    else:
        view1 = VIEW_1
        view2 = VIEW_2

    render_two_groups_pyvista(bb_pos, sc_pos, dims, out_png1, view1, off_screen=off_screen)
    render_two_groups_pyvista(bb_pos, sc_pos, dims, out_png2, view2, off_screen=off_screen)


def png_to_jpeg(png_path: Path, jpg_path: Path, quality=75, max_px=1400):
    """
    Convert to JPEG and optionally downscale so the largest side is max_px.
    """
    png_path = Path(png_path)
    jpg_path = Path(jpg_path)
    jpg_path.parent.mkdir(parents=True, exist_ok=True)

    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > max_px:
        scale = max_px / m
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    im.save(jpg_path, "JPEG", quality=quality, optimize=True, progressive=True)
    return jpg_path


def build_group_pdf_onepage(
    entries,
    out_pdf: Path,
    title: str,
    subtitle: str = "",
    max_cols: int = 3,
    margin_cm: float = 0.7,
    gutter_cm: float = 0.28,
    inner_gutter_cm: float = 0.18,
    caption_fs: int = 8,
    header_fs: int = 12,
    subtitle_fs: int = 8,
):
    """
    Single-page portrait contact sheet:
      - grid of peptides (<= max_cols columns)
      - each tile: caption + two views
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4
    page_h = page_h - 3 * cm
    canvas_obj = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))

    margin = margin_cm * cm
    gutter = gutter_cm * cm
    inner_gutter = inner_gutter_cm * cm

    header_h = 0.8 * cm if (title or subtitle) else 0.0

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - header_h

    n = len(entries)
    if n == 0:
        canvas_obj.setFont("Helvetica-Bold", header_fs)
        canvas_obj.drawString(margin, page_h - margin, title or "Empty group")
        canvas_obj.setFont("Helvetica", subtitle_fs)
        canvas_obj.drawString(margin, page_h - margin - 0.8 * cm, "No peptides found.")
        canvas_obj.save()
        return

    cols = min(max_cols, 3)
    if n <= 2:
        cols = n
    elif n <= 6:
        cols = min(2, cols)

    rows = int(ceil(n / cols))

    tile_w = (usable_w - (cols - 1) * gutter) / cols
    tile_h = (usable_h - (rows - 1) * gutter) / rows

    cap_h = 0.55 * cm

    y_top = page_h - margin
    if header_h > 0:
        canvas_obj.setFont("Helvetica-Bold", header_fs)
        canvas_obj.drawString(margin, y_top, title)
        if subtitle:
            canvas_obj.setFont("Helvetica", subtitle_fs)
            canvas_obj.drawString(margin, y_top - 0.60 * cm, subtitle)

    grid_top = page_h - margin - header_h

    def draw_placeholder(x, y, w, h, text="missing"):
        canvas_obj.rect(x, y, w, h, stroke=1, fill=0)
        canvas_obj.setFont("Helvetica", max(7, caption_fs - 1))
        canvas_obj.drawString(x + 0.15 * cm, y + h - 0.45 * cm, text)

    def sort_key(entry):
        ap = entry.get("ap")
        rm = entry.get("rmoi")
        apk = ap if ap is not None else -1e18
        rmk = rm if rm is not None else -1e18
        return (-apk, -rmk, str(entry.get("peptide", "")))

    entries_sorted = sorted(entries, key=sort_key)

    for idx, entry in enumerate(entries_sorted):
        row = idx // cols
        col = idx % cols

        tx = margin + col * (tile_w + gutter)
        ty_top = grid_top - row * (tile_h + gutter)
        ty = ty_top - tile_h

        peptide = str(entry.get("peptide", "NA"))
        ap = entry.get("ap")
        rm = entry.get("rmoi")
        cap = (
            f"{peptide} | AP: {(f'{ap:.3f}' if ap is not None else 'NA')} | "
            f"RMOI: {(f'{rm:.3f}' if rm is not None else 'NA')}"
        )
        canvas_obj.setFont("Helvetica", caption_fs)
        canvas_obj.drawString(tx + 0.05 * cm, ty_top - 0.45 * cm, cap[:120])

        img_area_h = tile_h - cap_h
        img_w = (tile_w - inner_gutter) / 2.0
        img_h = img_area_h

        img_y = ty
        img1 = Path(entry.get("img1_path") or "")
        img2 = Path(entry.get("img2_path") or "")

        def draw_img(path: Path, x, y, w, h):
            if (not path.exists()) or (path.stat().st_size < MIN_PNG_BYTES):
                draw_placeholder(x, y, w, h, "render failed")
                return
            try:
                jpg_path = Path(str(path)).with_suffix(".jpg")
                if (not jpg_path.exists()) or jpg_path.stat().st_mtime < path.stat().st_mtime:
                    png_to_jpeg(path, jpg_path, quality=75, max_px=1200)
                image_reader = ImageReader(str(jpg_path))
                canvas_obj.drawImage(
                    image_reader,
                    x,
                    y,
                    width=w,
                    height=h,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
            except Exception:
                draw_placeholder(x, y, w, h, "bad image")

        draw_img(img1, tx, img_y, img_w, img_h)
        draw_img(img2, tx + img_w + inner_gutter, img_y, img_w, img_h)

        canvas_obj.setLineWidth(0.4)
        canvas_obj.rect(tx, ty, tile_w, tile_h, stroke=1, fill=0)

    canvas_obj.save()


def resolve_runs_dir(base_dir: Path) -> Path:
    runs_dir = base_dir / "runs"
    return runs_dir if runs_dir.is_dir() else base_dir


def peptide_dir_for(base_dir: Path, run: str, cls: str, peptide: str) -> Path:
    runs_dir = resolve_runs_dir(base_dir)
    return runs_dir / run / cls / peptide


def cache_paths(peptide_dir: Path):
    cache_dir = peptide_dir / IMG_CACHE_DIRNAME
    return cache_dir / "view_1.png", cache_dir / "view_2.png"


def ensure_two_views(
    peptide_dir: Path,
    force=False,
    cutoff_nm=CUTOFF_NM,
    top_name=TOP_NAME,
    traj_name=TRAJ_NAME,
    off_screen=True,
):
    img1, img2 = cache_paths(peptide_dir)

    if (not force) and is_valid_png(img1) and is_valid_png(img2):
        return img1, img2

    img1.parent.mkdir(parents=True, exist_ok=True)
    render_two_views_for_peptide(
        peptide_dir,
        img1,
        img2,
        cutoff_nm=cutoff_nm,
        top_name=top_name,
        traj_name=traj_name,
        off_screen=off_screen,
    )
    return img1, img2


def parse_args() -> argparse.Namespace:
    base_dir = PEPMORPH_MAIN_ARTIFACTS_DIR
    outputs_dir = base_dir / "outputs"
    parser = argparse.ArgumentParser(description="Render PepMorph control contact sheets from analysis CSV.")
    parser.add_argument("--base-dir", type=str, default=str(base_dir))
    parser.add_argument("--csv-path", type=str, default=str(outputs_dir / "analysis_by_run.csv"))
    parser.add_argument("--output-dir", type=str, default=str(outputs_dir))
    parser.add_argument("--include-runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--include-classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--force-rerender", action="store_true")
    parser.add_argument("--cutoff-nm", type=float, default=CUTOFF_NM)
    parser.add_argument("--top-name", type=str, default=TOP_NAME)
    parser.add_argument("--traj-name", type=str, default=TRAJ_NAME)
    parser.add_argument("--use-filesystem", action="store_true")
    parser.add_argument("--on-screen", action="store_true", help="Disable off-screen rendering.")
    parser.add_argument("--max-cols", type=int, default=3)
    parser.add_argument("--caption-fs", type=int, default=8)
    parser.add_argument("--header-fs", type=int, default=12)
    parser.add_argument("--subtitle-fs", type=int, default=8)
    parser.add_argument("--margin-cm", type=float, default=0.7)
    parser.add_argument("--gutter-cm", type=float, default=0.28)
    parser.add_argument("--inner-gutter-cm", type=float, default=0.18)
    return parser.parse_args()


def init_pyvista(off_screen: bool) -> None:
    pv.set_jupyter_backend(None)
    if off_screen and platform.system() == "Linux":
        try:
            pv.start_xvfb(wait=0.1)
        except Exception as exc:
            raise RuntimeError(
                "Xvfb could not be started. Install it (e.g., apt-get install xvfb "
                "or conda-forge xorg-x11-server-xvfb) and retry."
            ) from exc


def get_peptides_for_group(
    df: pd.DataFrame,
    base_dir: Path,
    run: str,
    cls: str,
    use_filesystem: bool,
) -> list[str]:
    if use_filesystem:
        base = resolve_runs_dir(base_dir) / run / cls
        if not base.exists():
            return []
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    return df[(df["run"] == run) & (df["class"] == cls)]["peptide"].dropna().astype(str).unique().tolist()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_root = output_dir / "pdf_reports"
    out_root.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    off_screen = not args.on_screen
    init_pyvista(off_screen=off_screen)

    df = pd.read_csv(csv_path)
    required = ["peptide", "class", "run", "aggregation_propensity", "RMOI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["aggregation_propensity"] = pd.to_numeric(df["aggregation_propensity"], errors="coerce")
    df["RMOI"] = pd.to_numeric(df["RMOI"], errors="coerce")

    include_runs = args.include_runs or []
    include_classes = args.include_classes or []

    print("Runs:", include_runs)
    print("Classes:", include_classes)

    summary = []

    for run in include_runs:
        for cls in include_classes:
            sub = df[(df["run"] == run) & (df["class"] == cls)].copy()
            peptides = get_peptides_for_group(df, base_dir, run, cls, args.use_filesystem)

            entries = []
            print(f"\n=== {run} | {cls} | peptides={len(peptides)} ===")

            for pep in peptides:
                pdir = peptide_dir_for(base_dir, run, cls, pep)

                ap = None
                rm = None
                if not sub.empty and (sub["peptide"] == pep).any():
                    row = sub.loc[sub["peptide"] == pep].iloc[0]
                    ap = float(row["aggregation_propensity"]) if pd.notna(row["aggregation_propensity"]) else None
                    rm = float(row["RMOI"]) if pd.notna(row["RMOI"]) else None

                img1 = img2 = None
                try:
                    if pdir.exists():
                        img1, img2 = ensure_two_views(
                            pdir,
                            force=args.force_rerender,
                            cutoff_nm=args.cutoff_nm,
                            top_name=args.top_name,
                            traj_name=args.traj_name,
                            off_screen=off_screen,
                        )
                    else:
                        print(f"  [WARN] Missing directory: {pdir}")
                except Exception as exc:
                    print(f"  [WARN] Render failed for {pep} ({pdir}): {exc}")

                entries.append(
                    dict(
                        peptide=pep,
                        ap=ap,
                        rmoi=rm,
                        img1_path=str(img1) if img1 else None,
                        img2_path=str(img2) if img2 else None,
                    )
                )

            section = SECTION_LABELS.get(cls, cls)
            cohort = COHORT_LABELS.get(cls, cls)

            out_pdf = out_root / run / f"{cls}.pdf"

            build_group_pdf_onepage(
                entries,
                out_pdf,
                title=f"{run} | {section} | {cohort}",
                max_cols=args.max_cols,
                caption_fs=args.caption_fs,
                header_fs=args.header_fs,
                subtitle_fs=args.subtitle_fs,
                margin_cm=args.margin_cm,
                gutter_cm=args.gutter_cm,
                inner_gutter_cm=args.inner_gutter_cm,
            )

            summary.append((run, cls, len(entries), str(out_pdf)))

    print("\nDone. PDFs written:")
    for run, cls, n, path in summary:
        print(f"  {run:>5} | {cls:<20} | peptides={n:4d} | {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
