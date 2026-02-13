def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    h=len(X)
    w=len(X[0])
    h_output=((h-pool_size)//stride)+1
    w_output=((w-pool_size)//stride)+1
    output=[]
    for i in range(h_output):
        row = [] 
        for j in range(w_output):
            row_start=i*stride
            col_start =j*stride
            arr=[]
            for k in range(row_start,row_start+pool_size):
                for l in range(col_start,col_start+pool_size):
                    arr.append(X[k][l])
            max_value = max(arr)
            row.append(max_value)

        output.append(row)

    return output






