import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    row=np.array(A).shape[0]
    col=np.array(A).shape[1]
    A=np.array(A)
    arr=np.zeros((col,row))
    for i in range(row):
        for j in range(col):
            arr[j][i]=A[i][j]
    return arr
