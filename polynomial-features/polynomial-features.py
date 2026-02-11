def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    # 2,3-> 2^0 ,2^1,2^2

    # values=np.array(values)


    final_array=[]
    for i in range(len(values)):
        arr=[]
        for j in range(degree+1):
            k=values[i]
            l=k**j
            arr.append(l)
        final_array.append(arr)
    return final_array


