#! /usr/bin/env python3
'''
A neuron is a simple function, it get's two inputs, 
sum them with some weight and bias and finally pass it
through an activation function.
We'll use a sigmoid function as activation func.
'''

import numpy as np


def sigmoid(x):
    '''
        >>> raw = sigmoid(7)
        >>> round(raw, 4)
        0.9991
    '''
    return 1 / (1 + np.exp(-x))


class Neuron:
    '''
        this is a sample neuron:
        >>> weights = np.array([0,1])
        >>> bias = 4
        >>> n = Neuron(weights, bias)
        >>> x = np.array([2,3])
        >>> round(n.feedforward(x), 8)
        0.99908895
    '''

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class Network:
    '''
        lets test it:
            >>> network = Network()
            >>> x = np.array([2, 3])
            >>> network.feedforward(x)
            0.7216325609518421
    '''

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        ## now add hiden layers:
        ## all layers have input value of [2,3]
        ## so
        self.h1 = Neuron(weights, bias)  ## hidden layer 1
        self.h2 = Neuron(weights, bias)  ## hidden layer 2
        self.o1 = Neuron(weights, bias)  ## output

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


if __name__ == '__main__':
    import doctest
    doctest.testmod()
