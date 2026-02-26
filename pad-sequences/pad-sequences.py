import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is not None:
        for i in range(len(seqs)):
            if len(seqs[i])>max_len:
                seqs[i]=seqs[i][:max_len]
            else:
                short_by=max_len-len(seqs[i])
                seqs[i].extend(([pad_value]*short_by))
    else:
        max_len = max(len(seq) for seq in seqs)
        for i in range(len(seqs)):
            if len(seqs[i])<max_len:
                short_by=max_len-len(seqs[i])
                seqs[i].extend(([pad_value]*short_by))

            else:
                continue
    return np.array(seqs)
        
    
                
        