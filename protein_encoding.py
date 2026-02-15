# protein_encoding.py
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode_sequence(seq, max_len=1000):
    """
    Convert protein sequence to one-hot matrix.
    - seq: string of amino acids
    - max_len: fixed output length (truncate or pad)
    Returns: torch.Tensor of shape (max_len, 20)
    """
    import torch
    
    # Truncate or pad
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + 'X' * (max_len - len(seq))  # pad with 'X' (unknown)
    
    # Create zero matrix
    one_hot = torch.zeros((max_len, 20))
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            one_hot[i, aa_to_idx[aa]] = 1.0
        # else leave zeros for 'X' or non-standard
    return one_hot
# test_protein.py
from protein_encoding import one_hot_encode_sequence
seq = "ACDEFGHIKLMNPQRSTVWY" * 50  # 1000 length exactly
enc = one_hot_encode_sequence(seq)
print(enc.shape)  # Should be torch.Size([1000, 20])