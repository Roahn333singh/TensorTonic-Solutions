def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    step=chunk_size-overlap
    arr=[]
    for i in range(0,len(tokens),step):
        k=tokens[i:i+chunk_size]
        arr.append(k)
        if (i+chunk_size)>=len(tokens):
            break
    return arr


        







