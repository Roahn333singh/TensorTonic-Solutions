import numpy as np


def bert_embeddings(
    token_ids: np.ndarray,
    segment_ids: np.ndarray,
    token_embeddings: np.ndarray,
    position_embeddings: np.ndarray,
    segment_embeddings: np.ndarray,
) -> np.ndarray:
  """Returns the float64 BERT input embeddings with shape (B, S, H)."""
  _, seq_len = token_ids.shape
  position_ids = np.arange(seq_len)[None, :] 

  tok_emb = token_embeddings[token_ids]  
  pos_emb = position_embeddings[position_ids]  # shape (1, S, H) -> (B, S, H)
  seg_emb = segment_embeddings[segment_ids]  # shape (B, S, H)

  return (tok_emb + pos_emb + seg_emb).astype(np.float64)