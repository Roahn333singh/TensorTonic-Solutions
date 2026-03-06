import numpy as np
from collections import Counter
def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    # y=np.array(y)
    c=Counter(y)

 
    #counting no.of classes
    # uq=np.unique(y).count()

    lnth=len(y)
    k=0
    for i in c:
        pi=c[i]/(lnth)
        k+=-(pi*np.log2(pi))

    return float(k)
        
        

    
    