import numpy as np
import math
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    # 4,6
    # |s|c|s|c|s|c| x seq_len
    # i runs from 0 to d/2 - 1
    range_i= (d_model//2) -1
    denom_i=[]
    # arr=np.ndarray()
    enc_arr=[]
    # denominator calculation
    for i in range(range_i + 1):
        
        # for i=0 ,1,2
        # {0:x1,1:x2,2:x3}
        denom_i.append(1/(10000**((2*i)/d_model)))
                    
    for i in range(seq_length):
        # for pos 0,1,2,...
        # i-> (0,1,2)
        for j in range(len(denom_i)):
            s=math.sin(i*denom_i[j])
            c=math.cos(i*denom_i[j])
            enc_arr.append(s)
            enc_arr.append(c)
    # [1,1,1,1,1,1,]
    return np.array(enc_arr).reshape(seq_length,d_model)
            
                      
        

    
            
        
        
        
        
        
    