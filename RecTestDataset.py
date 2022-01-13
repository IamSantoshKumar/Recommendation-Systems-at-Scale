import torch
import numpy as np

class RecSysDataset:
    
    def __init__(self, usr_id, mov_id):
        self.usr_id = usr_id
        self.mov_id = mov_id
        
    def __len__(self):
        return len(self.usr_id)

    def __getitem__(self, idx):
        usr_id = self.usr_id[idx]
        mov_id = self.mov_id[idx]
        usr_id = torch.tensor(usr_id, dtype=torch.long)
        mov_id = torch.tensor(mov_id, dtype=torch.long)
        return usr_id, mov_id