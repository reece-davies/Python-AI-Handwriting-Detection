# Description: Just the code of a classifier on a neural network
# Full guide at: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

class NeuralNetwork:
    def __init__(self, x, y);
        self.x = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)