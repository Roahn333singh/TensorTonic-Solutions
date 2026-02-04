import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x=np.sort(x)
    arr=[]
    for i in q:
        if i==0:
            arr.append(x[0])
        elif i==100:
            arr.append(x[-1])
        else:
            k=(i/100)*(len(x)-1)
            min_indx=int(k)
            max_indx = min(min_indx + 1, len(x) - 1)
            cont=k-min_indx
            p=x[min_indx]+cont*(x[max_indx]-x[min_indx])

            arr.append(p)
    
    return np.array(arr)



    
    