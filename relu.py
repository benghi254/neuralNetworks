#Element wise operations applied to tensors

def naive_relu(x):
    #x is a2D Numpy tensor
    assert len(x.shape) == 2 

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i, j], 0)
    return x

def naive_add(x,y):
    # x and y are 2D Numpy tensors
    assert len(x.shape) == 2
    assert x.shape == y.shape

    #avoid ovewritting the input tensor
    X =x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]

    return x
