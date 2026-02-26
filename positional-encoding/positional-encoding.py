import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    # d_model -> like the length of positional embedding vector
    # seq_len  -> like the length of sentence
    # ['hai','my','name ','is','rohan']
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for j in range(d_model):
            power = (2 * (j // 2)) / d_model
            angle = pos / (base ** power)

            if j % 2 == 0:
                PE[pos, j] = np.sin(angle)
            else:
                PE[pos, j] = np.cos(angle)

    return PE
