# binding_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from mol_graph import mol_to_graph
from protein_encoding import one_hot_encode_sequence

class BindingDataset(Dataset):
    def __init__(self, dataframe):
        """
        dataframe: pandas DataFrame with columns 'smiles', 'seq', 'neg_log10_affinity_M'
        """
        self.df = dataframe.reset_index(drop=True)
        # Pre-filter invalid SMILES? We'll handle on-the-fly with None.
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        protein_seq = row['seq']
        affinity = row['neg_log10_affinity_M']
        
        # Convert SMILES to graph (includes target y)
        graph = mol_to_graph(smiles, affinity)
        if graph is None:
            # If invalid, return None – we'll filter in collate
            return None
        
        # One-hot encode protein
        prot_enc = one_hot_encode_sequence(protein_seq)
        
        return graph, prot_enc, affinity
    
# binding_dataset.py (continued)
from torch_geometric.data import Batch

def collate_fn(batch):
    # Remove None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # Return empty batch (handle gracefully)
        return None, None, None
    
    graphs, prot_encs, affinities = zip(*batch)
    
    # Batch graphs
    batched_graphs = Batch.from_data_list(graphs)
    
    # Stack protein tensors (they are already same shape due to padding)
    batched_prots = torch.stack(prot_encs, dim=0)  # [batch, max_len, 20]
    
    # Stack affinities
    batched_affs = torch.tensor(affinities, dtype=torch.float)
    
    return batched_graphs, batched_prots, batched_affs