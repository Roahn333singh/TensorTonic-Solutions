import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    c=Counter(x)
   
    mean_data=np.mean(x)
    median_data=np.median(x)

   
    mode_data=max(c.values())
    arr=[]
    for i ,j in c.items():
        if j==mode_data:
            arr.append(i)
    
    md=min(arr)


    return mean_data,median_data,md