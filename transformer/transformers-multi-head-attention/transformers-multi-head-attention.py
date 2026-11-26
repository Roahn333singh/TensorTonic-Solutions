import numpy as np
# import torch
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    h_dim=Q.shape[2]//num_heads
    B,T,H,D,H_d = Q.shape[0], Q.shape[1],num_heads, Q.shape[2] , h_dim

    # Q , K ,V
    # Q -> (1,3,4) -> ()
    Q_proj = Q @ W_q  # (B, T_q, D)
    K_proj = K @ W_k  # (B, T_k, D)
    V_proj = V @ W_v  # (B, T_k, D)
    
    Q_heads = Q_proj.reshape(B, T, num_heads, H_d).transpose(0, 2, 1, 3)  
    K_heads = K_proj.reshape(B, T, num_heads, H_d).transpose(0, 2, 1, 3) 
    V_heads = V_proj.reshape(B, T, num_heads, H_d).transpose(0, 2, 1, 3)  

    attn= (Q_heads @ np.transpose(K_heads,(0,1,3,2)))/np.sqrt(H_d)
    shifted = attn - np.max(attn, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True) 

    context = weights @ V_heads                             
    concat = context.transpose(0, 2, 1, 3).reshape(B, T, D)       
    return (concat @ W_o).astype(np.float64)                         