import numpy as np
import matplotlib.pyplot as plt
import time
import os

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    exp = np.exp(x)
    return exp/exp.sum(axis=1,keepdims=True)

def init_layers(batch_size,layer_sizes):
    hidden_layers=[np.empty((batch_size,layer_size)) for layer_size in layer_sizes]
    return hidden_layers


