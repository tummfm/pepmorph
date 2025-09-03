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
        self.max_length = max_length
        if amino_acids is None:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.amino_acids = amino_acids
        
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        L = random.randint(1, self.max_length)
        seq = ''.join(random.choices(self.amino_acids, k=L))
        cond = torch.zeros(self.condition_length, dtype=torch.float)
        # cond[0] = float(L) / self.max_length
        return seq, cond