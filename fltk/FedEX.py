import numpy as np

''' Initialization'''
k = 5   # Number of possible configurations (batch sizes)
configurations = [8, 16, 32, 64, 128]
theta = [1/k]*5   # Probability for each configuration


def FedEX():
    new_batch_size = 256
    return new_batch_size