import reg_class


for n in range(10,100,10):
    for i in range(10):
        
    x = np.sort((np.random.rand(n)))
    y = np.sort((np.random.rand(n)))
    
    x,y = np.meshgrid(x,y)
    
    deg = 5
    z = reg_class.FRANK(x.ravel(),y.ravel()) #+0.5*np.random.randn(n*n,1)
    X = polynomial_this(x,y,deg)
    print(X.shape)
    FRANK = regression(X,z)