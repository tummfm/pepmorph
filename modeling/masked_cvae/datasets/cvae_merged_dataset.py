import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class OnlinePretrainDataset(Dataset):
    def __init__(self, condition_length,
                 epoch_size=10000, max_length=10,
                 amino_acids=None):
        self.condition_length = condition_length
        self.max_length = max_length
        self.epoch_size = epoch_size
        if amino_acids is None:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.amino_acids = amino_acids

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        L = random.randint(1, self.max_length)
        seq = ''.join(random.choices(self.amino_acids, k=L))
        cond = torch.zeros(self.condition_length, dtype=torch.float)
        cond[0] = float(len(seq)) / self.max_length
        return seq, cond


FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

class CVAEAllDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 random_mask=False,
                 max_fasta_length: int = 20,
                 random_size: int = 15000):
        """
        df: your real CVAE DataFrame (must contain 'peptide' column + FEATURES)
        random_size: how many random peptides to inject per epoch
        """
        self.df = df.reset_index(drop=True)
        self.max_fasta_length = max_fasta_length
        self.random_mask = random_mask

        # build a random‐peptide dataset of size `random_size`
        self.random_size = random_size
        self.random_dataset = OnlinePretrainDataset(
            condition_length=len(FEATURES),
            epoch_size=random_size,
            max_length=max_fasta_length
        )

    def __len__(self):
        return len(self.df) + self.random_size

    def __getitem__(self, idx):
        if idx < len(self.df):
            row = self.df.iloc[idx]
            seq = row['peptide']

            cond = [
                row[feat] if feat in row and not pd.isna(row[feat])
                else 0.0
                for feat in FEATURES
            ]
            mask = [
                1.0 if feat in row and not pd.isna(row[feat])
                else 0.0
                for feat in FEATURES
            ]
            cond[0] = cond[0] / self.max_fasta_length

            cond_tensor = torch.tensor(cond, dtype=torch.float)
            mask_tensor = torch.tensor(mask, dtype=torch.float)

            if self.random_mask:
                mask_tensor *= torch.randint(2, size=(len(FEATURES),))
                if mask_tensor.sum() == 0:
                    mask_tensor[random.randint(0, len(FEATURES) - 1)] = 1.0

            return seq, cond_tensor, mask_tensor

        else:
            seq, cond_tensor = self.random_dataset[0]

            mask_tensor = torch.zeros(len(FEATURES), dtype=torch.float)
            mask_tensor[0] = 1.0
            return seq, cond_tensor, mask_tensor
