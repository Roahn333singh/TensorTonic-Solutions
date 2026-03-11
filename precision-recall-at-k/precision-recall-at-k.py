def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    # precision
    p=0
    for i in range(k):
        if recommended[i] in relevant:
            p+=1
    precision=p/k

            
    recall=p/len(relevant)

    return [precision,recall]

    
            
    