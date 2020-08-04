import numpy as np 

# X input feature set (training data set)
X =[[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):# weight matrice dimension
        #load batch
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # normalize by 0.10,(n_inputs, n_neurons) avoid tranpsose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

hidden_layer1 = Layer_Dense(4, 3) # Dimesions 3 x 4 must multiply with 3 x anything , resulting in 4 x anything matrix
hidden_layer2 = Layer_Dense(3, 3)

hidden_layer1.forward(X)
hidden_layer2.forward(hidden_layer1.output)
# should be a 3 by 4 matrice
print(hidden_layer2.output)
