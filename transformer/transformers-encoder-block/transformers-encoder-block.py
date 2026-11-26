import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    avg=np.mean(x,axis=-1,keepdims=True)
    var=np.var(x,axis=-1,keepdims=True)

    return (gamma * ((x-avg)/np.sqrt(var+eps)))+beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    head_dim=Q.shape[2]//num_heads
    B,T,H,H_d,D = Q.shape[0] , Q.shape[1] ,num_heads ,head_dim, Q.shape[2] 
    Q_=Q @ W_q
    K_=K @ W_k
    V_=V @ W_v

    Q_=Q_.reshape(B,T,H,H_d).transpose(0,2,1,3)
    K_=K_.reshape(B,T,H,H_d).transpose(0,2,1,3)
    V_=V_.reshape(B,T,H,H_d).transpose(0,2,1,3)

    attn= (Q_ @ K_.transpose(0,1,3,2))/np.sqrt(H_d)
    norm_attn=softmax(attn,axis=-1)
    context=norm_attn @ V_

    output=context.transpose(0,2,1,3).reshape(B,T,D)

    return output @ W_o
    
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    o1=np.maximum(0,x@W1 + b1)
    o2=o1 @ W2 + b2

    return o2
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    attn_layer = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_layer, gamma1, beta1)
    ffn_layer = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ffn_layer, gamma2, beta2)
    return x
    
