# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:05:16 2018

@author: bilal
"""

from random import uniform

class Network(object):
    
    def __init__(self, NETWORK_LAYER_SIZES):
        
        self.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES
        # self.bias = [np.random.randn(y, 1) for y in NETWORK_LAYER_SIZES[1:]]
        # self.weights = [np.random.randn(y, x)
        #               for x, y in zip(NETWORK_LAYER_SIZES[:-1], NETWORK_LAYER_SIZES[1:])]
        '''
        self.output = [np.random.randn(y, 1) for y in NETWORK_LAYER_SIZES[1:]]
        self.error_signal = [np.random.randn(y, 1) for y in NETWORK_LAYER_SIZES[1:]]
        self.output_derivative = [np.random.randn(y, 1) for y in NETWORK_LAYER_SIZES[1:]]'''
        
        self.NETWORK_SIZE = len(NETWORK_LAYER_SIZES)
        self.INPUT_SIZE = NETWORK_LAYER_SIZES[0]
        self.OUTPUT_SIZE =  NETWORK_LAYER_SIZES[self.NETWORK_SIZE - 1]
        
        
        self.output = self.NETWORK_SIZE * [None]
        self.error_signal = self.NETWORK_SIZE * [None]
        self.output_derivative = self.NETWORK_SIZE * [None]
        self.bias = self.NETWORK_SIZE * [None]
        for i in range(self.NETWORK_SIZE):
                self.output[i] = [0.0 for j in range(NETWORK_LAYER_SIZES[i])]
                self.error_signal[i] = [0.0 for j in range(NETWORK_LAYER_SIZES[i])]
                self.output_derivative[i] = [0.0 for j in range(NETWORK_LAYER_SIZES[i])]
                self.bias[i] = [uniform(0,1) for j in range(NETWORK_LAYER_SIZES[i])]
        
        self.weights = self.NETWORK_SIZE * [None]
        for i in range(1,self.NETWORK_SIZE):
            self.weights[i] = [ [uniform(0,1) for z in range(self.NETWORK_LAYER_SIZES[i-1])] for j in range(self.NETWORK_LAYER_SIZES[i])]
        
        
        
    def calculate(self, inputs):
        for i in range(self.INPUT_SIZE):
            self.output[0][i] = inputs[i]
        
        for layer in range(1, self.NETWORK_SIZE):
            for neuron in range(self.NETWORK_LAYER_SIZES[layer]):
                sum = self.bias[layer][neuron]
                
                for prevNeuron in range(self.NETWORK_LAYER_SIZES[layer - 1]):
                    sum += self.output[layer - 1][prevNeuron] * self.weights[layer][neuron][prevNeuron]

                self.output[layer][neuron] = leaky_ReLU(sum)
                self.output_derivative[layer][neuron] = leaky_ReLU_derivative(sum)
                # self.output[layer][neuron] = sigmoid(sum)
                # self.output_derivative[layer][neuron] = sigmoid_derivative(sum)
    
        return self.output[self.NETWORK_SIZE - 1]
    
    def backPropError(self, target):
        for neuron in range(self.NETWORK_LAYER_SIZES[self.NETWORK_SIZE - 1]):
            self.error_signal[self.NETWORK_SIZE - 1][neuron] = (self.output[self.NETWORK_SIZE - 1][neuron] - target[neuron]) * self.output_derivative[self.NETWORK_SIZE - 1][neuron];
        for layer in range(self.NETWORK_SIZE - 2, 0,-1):
            for neuron in range(self.NETWORK_LAYER_SIZES[layer]):
                sum = 0
                for nextNeuron in range(self.NETWORK_LAYER_SIZES[layer+1]):
                    sum += self.weights[layer +1][nextNeuron][neuron] * self.error_signal[layer +1 ][nextNeuron]
                self.error_signal[layer][neuron] = sum * self.output_derivative[layer][neuron];    
     
    def updateWeights(self, eta):
        for layer in range(1,self.NETWORK_SIZE):
            for neuron in range( self.NETWORK_LAYER_SIZES[layer]):
                delta = -eta * self.error_signal[layer][neuron]
                self.bias[layer][neuron] += delta
                for prevNeuron in range(self.NETWORK_LAYER_SIZES[layer-1]):
                    self.weights[layer][neuron][prevNeuron] += delta * self.output[layer - 1][prevNeuron]
            
    
    
    
    def train(self, inputs, target, eta):
        # calculate the each output of each neuron.
        self.calculate(inputs)

        # calculate error signals
        self.backPropError(target)
        
        # update weights.
        self.updateWeights(eta)


        
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
def sigmoid_derivative(x): 
    return sigmoid(x) * (1 - sigmoid(x))

def leaky_ReLU(x):
    if(x <= 0):
        return 0.01 * x
    else:
        return x

def leaky_ReLU_derivative(x):
    if(x <= 0):
        return 0.01
    else:
        return 1
        
'''
x = [1,2,3,4,5,6,7,8]

NETWORK_LAYER_SIZES = [2,3,1]
NETWORK_SIZE = len(NETWORK_LAYER_SIZES)

output = NETWORK_SIZE * [None]
for i in range(NETWORK_SIZE):
    output[i] = [0.0 for j in range(NETWORK_LAYER_SIZES[i])]
    
www = NETWORK_SIZE * [None]
for i in range(1,NETWORK_SIZE):
    www[i] = [ [0.0 for z in range(NETWORK_LAYER_SIZES[i-1])] for j in range(NETWORK_LAYER_SIZES[i])]
print(www)        

weights = [np.random.randn(y, x) for x, y in zip(NETWORK_LAYER_SIZES[:-1], NETWORK_LAYER_SIZES[1:])]

print(weights)  '''
        

