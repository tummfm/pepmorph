# PepMorph: Morphology-Specific Peptide Discovery via Masked Conditional Generative Modeling

This repository implements the **PepMorph** pipeline for the paper *Morphology-Specific Peptide Discovery via Masked Conditional Generative Modeling* - Costa & Zavadlav, (2025). It provides a conditional, mask-aware CVAE for peptide generation and utilities to train, validate, and compare generated sequences against morphology targets.

---

## Repository structure

```
├── dataset/
│   ├── figs/                           # resulting visualizations
│   ├── *.ipynb                          # notebooks for data evaluation
│   ├── merged_all_no_norm.csv          # merged dataset of peptide sequences and descriptors (unnormalized)
│   └── merged_all.csv                  # merged dataset of peptide sequences and descriptors
├── descriptor_calc/
│   ├── pepfold_pipeline/                  # example files for MD setup
│   │   ├── generate_pep_dataset.py     # generate dataset of descriptors from pdb structures
│   │   └── test_sequences/               # generated descriptor validation
│   ├── gen.py                    # generate 3D structures of peptides in peptides.fst
│   └── peptides.fst              # list of peptides to generate conformations for
├── md_sims/
│   ├── example/                  # example files for MD setup
│   └── script_*.csv              # scripts for MD simulation
├── modeling/
│   ├── ap_model/
│   │   ├── datasets/             # dataloader
│   │   ├── models/               # model architecture
│   │   └── ap_sa_pred.ipynb      # notebook to train and validate the AP classifier and regressor
│   ├── masked_cvae  
│   │   ├── datasets/             # dataloader
│   │   ├── models/               # model architecture
│   │   ├── train.py              # training script for the masked CVAE model
│   │   └── utils.py              # utilities script
│   └── validation/
│       ├── figs/                # resulting plots
│       ├── gen_peptides/        # generated peptides in screening for morphology
│       ├── results/             # validation metrics results
│       └── *.ipynb              # notebooks for model validation
├── README.md
```

---

## Citation

If you use PepMorph, please cite:

```bibtex
@misc{costa2025morphologyspecificpeptidediscoverymasked,
      title={Morphology-Specific Peptide Discovery via Masked Conditional Generative Modeling}, 
      author={Nuno Costa and Julija Zavadlav},
      year={2025},
      eprint={2509.02060},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2509.02060}, 
}
```
---

## Contact

For questions regarding implementation, please send an email to nuno.costa@tum.de
