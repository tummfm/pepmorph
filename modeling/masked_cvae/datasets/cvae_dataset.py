import pandas as pd
import torch
from torch.utils.data import Dataset
import random

FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

def generate_cond_mask_vectors(**kwargs):
    condition = torch.zeros(len(FEATURES), dtype=torch.float)
    mask = torch.zeros(len(FEATURES), dtype=torch.float)

    for idx, feature in enumerate(FEATURES):
        value = kwargs.get(feature, None)
        if value is not None:
            condition[idx] = value
            mask[idx] = 1.0
    return condition, mask

class CVAEDataset(Dataset):
    def __init__(self, df, random_mask=False, max_fasta_length = 20):
        self.df = df.reset_index(drop=True)
        self.max_fasta_length = max_fasta_length
        self.random_mask = random_mask
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['peptide']
        
        cond = [row[feat] if feat in row and not pd.isna(row[feat]) else 0 for feat in FEATURES]
        mask = [1 if feat in row and not pd.isna(row[feat]) else 0 for feat in FEATURES]

        cond[0] = cond[0] / self.max_fasta_length

        cond_tensor = torch.tensor(cond, dtype=torch.float)
        mask_tensor = torch.tensor(mask, dtype=torch.float)

        if self.random_mask:
            mask_tensor *= torch.randint(2, size=(len(FEATURES),))
            if mask_tensor.sum() == 0:
                mask_tensor[random.randint(0, len(FEATURES) - 1)] = 1.0

        return seq, cond_tensor, mask_tensor
