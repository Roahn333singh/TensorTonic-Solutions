import torch
import torch.nn.functional as F
import math
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    return (torch.softmax((Q @ torch.transpose(K,1,2))/math.sqrt(K.shape[2]),-1)) @ V