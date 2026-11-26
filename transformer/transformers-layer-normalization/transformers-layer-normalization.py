import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    avg=np.mean(x,axis=-1,keepdims=True)
    var=np.var(x,axis=-1,keepdims=True)
    # gamma=gamma.reshape(1,gamma.shape[0])
    return (gamma * ((x-avg)/np.sqrt(var+eps)))+beta