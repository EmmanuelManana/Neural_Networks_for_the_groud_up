
import numpy as np 

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

#simple hidden layer
layer_output =[]
for neuron_weight, neuron_bias in zip(weights, biases): # associate each bias with its weights ie. [([0.2, 0.8, -0.5, 1.0], 2.0), ...]
    #dot product: input · weight = |weight| × |implut| × cos(θ) = suum of the products of weight and inputs
    neuron_output = 0
    for neuron_input, n_weight in zip(inputs, neuron_weight):
        neuron_output += neuron_input*n_weight
    #add the bias
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

#doing the exactly above using numpy
l_output = np.dot(weights, inputs) + biases
print(l_output)
   