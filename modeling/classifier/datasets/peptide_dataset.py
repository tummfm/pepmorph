from torch.utils.data import Dataset

class PeptidePredictorDataset(Dataset):
    def __init__(self, df, task='classification'):
        self.df = df.reset_index(drop=True)
        self.task = task
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['peptide']
        if self.task == 'classification':
            label = row['is_assembled']
            return sequence, label
        elif self.task == 'regression':
            ap_value = row['ap']
            return sequence, ap_value