# mol_graph.py (updated)
from rdkit import Chem
import torch
from torch_geometric.data import Data

def mol_to_graph(smiles, affinity=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: atomic numbers
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)

    # Edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    if affinity is not None:
        data.y = torch.tensor([affinity], dtype=torch.float)
    return data