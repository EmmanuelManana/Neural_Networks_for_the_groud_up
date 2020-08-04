
import numpy as np 

inputs =[[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

inputs_1 = [1.0, 2.0, 3.0, 2.5]
           

weights = [[0.2, 0.8, -0.5, 1.0], # 3 by 4
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.73]]

biases = [2.0, 3.0, 0.5]
biases2 = [2, 3, 0.5]

#simple hidden layer
layer_output =[]
for neuron_weight, neuron_bias in zip(weights, biases): # associate each bias with its weights ie. [([0.2, 0.8, -0.5, 1.0], 2.0), ...]
    #dot product: input · weight = |weight| × |input| × cos(θ) = suum of the products of weight and inputs
    neuron_output = 0
    for neuron_input, n_weight in zip(inputs_1, neuron_weight):
        neuron_output += neuron_input*n_weight
    #add the bias
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

#doing the exactly above using numpy
first_hidden_layer = np.dot(inputs, np.array(weights).T) + biases #first hidden layer
layer_output = np.dot(first_hidden_layer, np.array(weights2).T) + biases2
print(layer_output)
   