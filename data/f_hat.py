def f_hat(x):
    #x : torch tensor shape (2,)
    #returns dx/dt in physical units
    
    x_n = (x - X_mean) / X_std

    with torch.no_grad():
        y_n = model(x_n)

    y = y_n * Y_std + Y_mean
    return y
