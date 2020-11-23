class Network(object):
    
    def _init_(self, sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def sigmond(z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self,a):

        for b,w in zip(self.biases, self.weights):
            a=sigmond(np.dot(w,a)+b)
            return a





