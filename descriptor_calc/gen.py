import os
import sys
import subprocess
import argparse
import re
from time import sleep
from Bio import SeqIO
import multiprocessing

def safe_folder_name(name):
    """Generate a safe folder name (alphanumeric and underscore only)."""
    return re.sub(r'\W+', '_', name)

def run_command(cmd, cwd):
    """Run a shell command in the given cwd; print command and return exit code."""
    print("\nRunning command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print("Error: command failed with code", result.returncode)
    return result.returncode

def process_chunk(chunk_records, chunk_index, start_index, sa_dir, label_prefix):
    """
    Process one chunk:
      - Write a temporary FASTA file for the chunk.
      - Run PyPPP3ListExec in a container with its own /tmp (using --tmpfs)
        and a unique socket (using --socket).
      - If the command fails, retry a few times.
      - Rename the generated SA profile files to a global naming scheme.
    """
    chunk_dir = os.path.join(sa_dir, f"chunk_{chunk_index}")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_fasta = os.path.join(chunk_dir, "chunk.fst")
    with open(chunk_fasta, "w") as fout:
        for rec in chunk_records:
            SeqIO.write(rec, fout, "fasta")
    
    pwd = os.getcwd()
    uid = str(os.getuid())
    gid = str(os.getgid())
    chunk_label = f"{label_prefix}_chunk{chunk_index}"
    
    cmd = [
        "docker", "run", "-it", "--rm",
        "-v", f"{pwd}:{pwd}:z",
        "--tmpfs", "/tmp",
        "-u", f"{uid}:{gid}",
        "-w", chunk_dir,
        "pyppp3-light", "PyPPP3ListExec", "20",
        "--iSeqList", "chunk.fst",
        "-l", chunk_label,
    ]
    
    max_retries = 1
    for attempt in range(max_retries):
        ret = run_command(cmd, cwd=chunk_dir)
        if ret == 0:
            break
        else:
            print(f"Chunk {chunk_index}: Attempt {attempt+1}/{max_retries} failed. Retrying in 2 seconds...")
            sleep(2)
    if ret != 0:
        print(f"SA profile generation failed for chunk {chunk_index} after {max_retries} attempts.")
        return None
    else:
        print(f"Chunk {chunk_index} SA profiles generated.")
    
    for file in os.listdir(chunk_dir):
        if file.endswith(".svmi8.27.prob"):
            m = re.search(rf"{re.escape(chunk_label)}_(\d+)\.svmi8\.27\.prob", file)
            if m:
                local_index = int(m.group(1))
                global_index = start_index + local_index - 1
                new_name = f"{label_prefix}_{global_index}.svmi8.27.prob"
                os.rename(os.path.join(chunk_dir, file), os.path.join(sa_dir, new_name))
    return True

def stage1_generate_sa_profiles_parallel(input_fasta, sa_dir, label_prefix, sa_workers):
    """
    Stage 1 (accelerated):
      - Reads the input FASTA.
      - Splits it into 'sa_workers' chunks.
      - Runs PyPPP3ListExec concurrently for each chunk.
      - Renames each generated SA profile to a global scheme:
        <label_prefix>_globalIndex.svmi8.27.prob.
    """
    os.makedirs(sa_dir, exist_ok=True)
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total = len(records)
    print(f"Total sequences: {total}")

    q, r = divmod(total, sa_workers)
    chunks = []
    start = 0
    for i in range(sa_workers):
        size = q + (1 if i < r else 0)
        chunks.append(records[start:start+size])
        start += size

    pool_args = []
    for i, chunk in enumerate(chunks):
        start_index = sum(len(c) for c in chunks[:i]) + 1
        pool_args.append((chunk, i + 1, start_index, sa_dir, label_prefix))

    with multiprocessing.Pool(processes=sa_workers) as pool:
        results = pool.starmap(process_chunk, pool_args)

    if any(r is None for r in results):
        print("One or more SA profile chunks failed.")
        sys.exit(1)
    print("All SA profiles generated in parallel.")
    sleep(5)

def split_fasta_to_seq(input_fasta, seq_dir, label_prefix):
    """
    Split the input FASTA into individual .seq files (one per peptide).
    Returns a list of tuples: (index, peptide_id, peptide_folder, seq_filepath).
    Global peptide IDs will be in the form <label_prefix>_globalIndex.
    """
    os.makedirs(seq_dir, exist_ok=True)
    entries = []
    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta"), start=1):
        peptide_id = f"{label_prefix}_{i}"
        peptide_folder = os.path.join(seq_dir, peptide_id)
        os.makedirs(peptide_folder, exist_ok=True)
        seq_filename = f"{peptide_id}.seq"
        seq_filepath = os.path.join(peptide_folder, seq_filename)
        if not f"{str(record.seq).strip()}\n":
            continue
        with open(seq_filepath, "w") as f:
            f.write(f"{str(record.seq).strip()}\n")
        entries.append( (i, peptide_id, peptide_folder, seq_filepath) )
    return entries

def process_peptide_structure(entry, sa_dir, structures_dir, label_prefix, max_retries=1):
    """
    Process a single peptide:
      - Copy its .seq file into a structure folder.
      - Run pepfold-core using the matching SA profile.
      - If the command fails (exit code 137) or no final output files are generated,
        retry with reduced simulation parameters.
      - Move final outputs (e.g., bestmodels.pdb, .trj) into a 'final' subfolder,
        and clean up intermediate files.
    """
    idx, peptide_id, seq_folder, seq_filepath = entry
    print(f"\n[Worker] Processing {peptide_id} (global index {idx})")
    pwd = os.getcwd()
    uid = str(os.getuid())
    gid = str(os.getgid())
    
    sa_profile = os.path.join(sa_dir, f"{label_prefix}_{idx}.svmi8.27.prob")
    if not os.path.exists(sa_profile) or os.path.getsize(sa_profile) == 0:
        print(f"[Worker] SA profile not found or empty for {peptide_id} at {sa_profile}. Skipping.")
        return
    
    structure_folder = os.path.join(structures_dir, peptide_id)
    pdb_file = os.path.join(structure_folder, f"{peptide_id}-bestmodel.pdb")

    if os.path.isdir(structure_folder) and os.path.exists(pdb_file):
        print(f"[Worker] {pdb_file} already exists. Skipping processing for {peptide_id}.")
        return
    
    os.makedirs(structure_folder, exist_ok=True)
    structure_seq = os.path.join(structure_folder, os.path.basename(seq_filepath))
    subprocess.run(["cp", seq_filepath, structure_seq])
    
    nSim = 50
    nc = 20
    attempt = 0
    success = False
    
    while attempt <= max_retries and not success:
        print(f"[Worker] Running pepfold-core for {peptide_id} (attempt {attempt+1}) with --nSim {nSim} and --nc {nc}")
        cmd = [
            "docker", "run", "-it", "--rm",
            "-v", f"{pwd}:{pwd}:z",
            "-u", f"{uid}:{gid}",
            "-w", structure_folder,
            "pepfold-core", "CppBuilder",
            "--objective", "sopep",
            "--generator", "fbt",
            "--action", "pepbuild",
            "--sOPEP_version", "2",
            "--nSim", str(nSim),
            "--iSeq", structure_seq,
            "--unbias",
            "--iPrf", sa_profile,
            "--label", peptide_id,
            "--nc", str(nc),
        ]
        
        ret = run_command(cmd, cwd=structure_folder)
        output_files = [f for f in os.listdir(structure_folder) if f.endswith("bestmodels.pdb") or f.endswith(".trj")]
        if ret == 0 and output_files:
            success = True
            print(f"[Worker] pepfold-core succeeded for {peptide_id} on attempt {attempt+1}.")
        else:
            print(f"[Worker] pepfold-core failed for {peptide_id} (exit code {ret}) or produced no outputs.")
            attempt += 1

            if ret == 137 or not output_files:
                new_nc = max(10, nc - 5)
                if new_nc != nc:
                    print(f"[Worker] Reducing --nc from {nc} to {new_nc} for {peptide_id}.")
                    nc = new_nc
            sleep(5)

def stage2_build_structures(entries, sa_dir, structures_dir, label_prefix, num_workers):
    """
    Use a multiprocessing pool to run structure generation for each peptide in parallel.
    """
    pool_args = []
    for entry in entries:
        pool_args.append( (entry, sa_dir, structures_dir, label_prefix) )
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_peptide_structure, pool_args)

def main():
    parser = argparse.ArgumentParser(
        description="Accelerated PEP-FOLD pipeline with parallel SA profile generation and parallel structure building."
    )
    parser.add_argument("input_fasta", help="Input FASTA file containing all peptides.")
    parser.add_argument("--output-dir", default="pepfold_pipeline",
                        help="Base output directory.")
    parser.add_argument("--label-prefix", default="peptide",
                        help="Global label prefix for SA profiles and peptide IDs (e.g., 'peptide').")
    parser.add_argument("--sa-workers", type=int, default=4,
                        help="Number of parallel workers for SA profile generation (stage 1).")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for structure generation (stage 2).")
    parser.add_argument('--run-sa-gen', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-3d-gen', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    base_out = os.path.abspath(args.output_dir)
    sa_dir = os.path.join(base_out, "sa_profiles")
    seq_dir = os.path.join(base_out, "seq_files")
    structures_dir = os.path.join(base_out, "peptide_structures")
    
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(sa_dir, exist_ok=True)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)
    
    if args.run_sa_gen:
        print("Stage 1: Generating SA profiles in parallel using PyPPP3ListExec...")
        stage1_generate_sa_profiles_parallel(args.input_fasta, sa_dir, args.label_prefix, args.sa_workers)

    print("\nStage 2: Splitting FASTA and building 3D structures in parallel...")
    entries = split_fasta_to_seq(args.input_fasta, seq_dir, args.label_prefix)
    
    if args.run_3d_gen:
        print(f"Found {len(entries)} peptides in the FASTA file.")
        stage2_build_structures(entries, sa_dir, structures_dir, args.label_prefix, args.num_workers)
    
    print("\nPEP-FOLD pipeline processing completed.")

if __name__ == "__main__":
    main()
