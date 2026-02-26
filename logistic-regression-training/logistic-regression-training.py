import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    x_shape=X.shape[1]
    w=np.zeros(x_shape)
    b=0.0
    for i in range(steps):

        z = X @ w + b
        p=_sigmoid(z)

        grad_w=(1/x_shape)*(X.T@(p-y))
        grad_b=(1/x_shape)*(np.sum(p-y))

        w=w-lr*(grad_w)
        b=b-lr*(grad_b)

    return w,b
        
        

        
        
        
        
    