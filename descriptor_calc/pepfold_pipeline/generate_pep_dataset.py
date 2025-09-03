
import os
import glob
import csv
import math
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, DSSP, Polypeptide, PDBIO
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Eisenberg hydrophobicity scale dictionary.
HYDRO_SCALE = {
    'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
    'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
    'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
    'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
}

def compute_beta_sheet_fraction(pdb_filename, structure, dssp_exe='mkdssp'):
    """Compute the fraction of residues in beta-strand using DSSP."""
    model = structure[0]
    try:
        dssp_obj = DSSP(model, pdb_filename, dssp=dssp_exe)
        dssp = dict(dssp_obj)
        total = len(dssp)
        beta_count = sum(1 for key in dssp if dssp[key][2] == 'E')
        return beta_count / total if total > 0 else 0.0
    except Exception as e:
        print(f"Warning: Could not run DSSP on {pdb_filename}: {e}")
        return None

def compute_hydrophobic_moment(sequence, structure):
    """
    Compute a simplified hydrophobic moment.
    For each residue, if a Cβ atom is available, use the vector CA -> CB.
    Otherwise (e.g. for Glycine), use an arbitrary fixed direction.
    Sum the product of the hydrophobicity value and the unit vector,
    then return the magnitude normalized by sequence length.
    """
    parser = PDBParser(QUIET=True)
    model = structure[0]
    chain = list(model.get_chains())[0]
    cum_vec = np.array([0.0, 0.0, 0.0])
    count = 0
    for residue in chain:
        resname = residue.get_resname().strip()
        # Convert three-letter code to one-letter code if possible.
        try:
            index = Polypeptide.three_to_index(resname)
            one_letter = Polypeptide.index_to_one(index)
        except KeyError:
            continue
        if one_letter not in HYDRO_SCALE:
            continue
        h_val = HYDRO_SCALE[one_letter]
        # Get CA coordinate.
        if 'CA' not in residue:
            continue
        ca = np.array(residue['CA'].get_coord())
        # Try to get CB direction.
        if 'CB' in residue:
            cb = np.array(residue['CB'].get_coord())
            vec = cb - ca
        else:
            # For glycine or missing CB, define a default vector:
            vec = np.array([1.0, 0.0, 0.0])
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        unit_vec = vec / norm
        cum_vec += h_val * unit_vec
        count += 1
    if count == 0:
        return 0.0
    # Normalize by the number of residues considered.
    return np.linalg.norm(cum_vec) / count

def compute_net_charge(sequence):
    """
    Compute the net charge of the peptide based on its sequence.
    Assume pH 7: +1 for K,R; -1 for D,E.
    """
    charge = 0
    for aa in sequence:
        if aa.upper() in ['K', 'R']:
            charge += 1
        elif aa.upper() in ['D', 'E']:
            charge -= 1
    return charge

def process_peptide_folder(folder_path):
    """
    Process a single peptide folder.
    Assumes files:
      - <folder_name>.seq : FASTA file containing one sequence.
      - <folder_name>-bestmodel.pdb : The best model PDB file.
    Returns a dictionary with peptide ID and computed metrics.
    """
    peptide_id = os.path.basename(folder_path)
    # Locate sequence file (*.seq)
    seq_files = glob.glob(os.path.join(folder_path, "*.seq"))
    if not seq_files:
        print(f"No FASTA file found in {folder_path}. Skipping.")
        return None
    # Assume the first FASTA file is the sequence.
    with open(seq_files[0], 'r') as fasta_file:
        sequence = fasta_file.read().strip()

    # Locate bestmodel PDB file.
    pdb_pattern = os.path.join(folder_path, f"{peptide_id}-bestmodel.pdb")
    pdb_files = glob.glob(pdb_pattern)
    if not pdb_files:
        print(f"No best model PDB file found in {folder_path} using pattern {pdb_pattern}. Skipping.")
        return None
    pdb_file = pdb_files[0]
    clean_pdb_file = os.path.join(folder_path, f"{peptide_id}-bestmodel-clean.pdb")

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(peptide_id, pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return None

    try:
        io = PDBIO()
        io.set_structure(structure)
        io.save(clean_pdb_file)
        structure = parser.get_structure(peptide_id, clean_pdb_file)
    except Exception as e:
        print(f"Error parsing or saving clean PDB file {clean_pdb_file}: {e}")
        return None

    # Compute metrics.
    beta_sheet_fraction = compute_beta_sheet_fraction(clean_pdb_file, structure)
    hydrophobic_moment = compute_hydrophobic_moment(sequence, structure)
    net_charge = compute_net_charge(sequence)

    return {
        "peptide_id": peptide_id,
        "sequence": sequence,
        "beta_sheet_fraction": beta_sheet_fraction,
        "hydrophobic_moment": hydrophobic_moment,
        "net_charge": net_charge,
    }

def main():
    peptide_folders = [d for d in glob.glob("./peptide_structures/peptide_*") if os.path.isdir(d)]
    print(f"Found {len(peptide_folders)} peptide folders.")

    results = []
    with ProcessPoolExecutor() as executor:
        future_to_folder = {executor.submit(process_peptide_folder, folder): folder for folder in peptide_folders}
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                peptide_data = future.result()
                if peptide_data is not None:
                    results.append(peptide_data)
                    print(f'Peptide {peptide_data['peptide_id']} finished.')
            except Exception as exc:
                print(f"{folder} generated an exception: {exc}")

    output_file = "peptide_metrics.csv"
    fieldnames = [
        "peptide_id", "sequence", "beta_sheet_fraction",
        "hydrophobic_moment", "net_charge",
    ]
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)
    print(f"Dataset written to {output_file}")

if __name__ == "__main__":
    main()

