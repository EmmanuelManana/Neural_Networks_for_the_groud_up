import numpy as np 

np.random.seed(0)
# # X input feature set (training data set)
# X =[[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

#create dummp data
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number + 1)* 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


X, y = create_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):# weight matrice dimension
        #load batch
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # normalize by 0.10,(n_inputs, n_neurons) avoid tranpsose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLu: #Rectified linear unit
    def __init__(self):
        pass
    # ReLu ()
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        

hidden_layer1 = Layer_Dense(2, 5) # Dimesions 3 x 4 must multiply with 3 x anything , resulting in 3 x anything matrix
Activation = Activation_ReLu()

hidden_layer1.forward(X)

Activation.forward(hidden_layer1.output)
print(hidden_layer1.output)
print(Activation.output)
