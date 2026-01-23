#!/usr/bin/env python3
import argparse, json
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.sparse import csr_matrix, csgraph

# Martini 3 unscaled bead radii (nm) and default masses (amu)
RADII_NM   = {"R":0.264, "S":0.230, "T":0.191}
DEFAULT_M  = {"R":72.0,  "S":54.0,  "T":36.0}

def default_radius(atom_name):
    c = atom_name[0].upper()
    return RADII_NM.get(c, RADII_NM["R"])

def default_mass(atom_name):
    c = atom_name[0].upper()
    return DEFAULT_M.get(c, DEFAULT_M["R"])

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute inertia tensor of the largest CG‐bead cluster"
    )
    p.add_argument("-g","--gro",    required=True, help=".gro topology")
    p.add_argument("-x","--xtc",    required=True, help=".xtc trajectory")
    p.add_argument("-c","--cutoff", required=True, type=float,
                   help="surface‐to‐surface cutoff (nm)")
    p.add_argument("-r","--radii", default=None,
                   help="optional JSON atom_name→radius (nm) overrides")
    p.add_argument("-m","--masses", default=None,
                   help="optional JSON atom_name→mass (amu) overrides")
    p.add_argument("-o","--out",    default=None,
                   help="(unused) kept for API compatibility")
    return p.parse_args()

def minimal_image_unwrap(pts, box):
    L = np.array(box[:3])  # box lengths (Å)
    ref = pts[0]
    unwrapped = np.zeros_like(pts)
    for i, p in enumerate(pts):
        d = p - ref
        d -= np.round(d / L) * L
        unwrapped[i] = ref + d
    return unwrapped

def inertia_tensor(pts, masses):
    I = np.zeros((3,3))
    for r, m in zip(pts, masses):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    return I

def main():
    args = parse_args()

    # 1) Load last frame
    u = mda.Universe(args.gro, args.xtc, refresh_offsets=True)
    u.trajectory[-1]
    coords = u.atoms.positions      # Å
    box    = u.dimensions           # [Lx Ly Lz α β γ] in Å & deg

    # 2) Radii (Å)
    radii_nm_map = json.load(open(args.radii)) if args.radii else {}
    radii = []
    for atom in u.atoms:
        r_nm = radii_nm_map.get(atom.name, default_radius(atom.name))
        radii.append(r_nm * 10.0)
    radii = np.array(radii)

    # 3) Cluster using PBC distances
    cutoff = args.cutoff * 10.0
    D = distance_array(coords, coords, box=box)
    Rsum = radii[:,None] + radii[None,:]
    adj = (D - Rsum) < cutoff
    np.fill_diagonal(adj, False)
    graph = csr_matrix(adj)
    nclus, labels = csgraph.connected_components(graph, directed=False)

    # 4) Pick largest cluster
    sizes  = np.bincount(labels)
    largest= np.argmax(sizes)
    idx    = np.where(labels == largest)[0]
    print(f"→ {nclus} clusters found; largest is #{largest} with {sizes[largest]} beads")

    # 5) Masses (amu)
    mass_map = json.load(open(args.masses)) if args.masses else {}
    masses = []
    for atom in u.atoms[idx]:
        m = mass_map.get(atom.name, default_mass(atom.name))
        masses.append(m)
    masses = np.array(masses)

    # 6) Unwrap & center at COM
    pts = coords[idx]
    pts = minimal_image_unwrap(pts, box)
    M   = masses.sum()
    com = np.dot(masses, pts) / M
    pts_centered = pts - com

    # 7) Inertia tensor & eigen decomposition
    I       = inertia_tensor(pts_centered, masses)
    evals, evecs = np.linalg.eigh(I)

    # 9) Report
    np.set_printoptions(precision=4, suppress=True)
    print("\nInertia tensor (amu·Å²):\n", I)
    print("\nPrincipal moments (amu·Å²):", evals)
    print("\nPrincipal axes (columns):\n", evecs)
    print("\nRatio of principal moments:", evals[0] / evals[2])

if __name__ == "__main__":
    main()
