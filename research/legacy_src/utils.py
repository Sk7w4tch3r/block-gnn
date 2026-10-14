import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import functools


def image_to_adj(image, kernel=None, block=False):
    
    w, h = image.shape
    adj = torch.zeros(w * h, w * h)
    
    kernel_size = 3
    if kernel is not None:
        kernel_size = kernel.shape[0]

    assert kernel_size % 2 == 1, "Kernel size must be odd"

    for i in range(w):
        for j in range(h):
            kernel_counter = 0
            interval = kernel_size//2
            for ii in range(i - interval, i + interval + 1):
                for jj in range(j - interval, j + interval + 1):
                    if 0 <= ii < w and 0 <= jj < h:
                        if kernel is None:
                            adj[i * h + j, ii * h + jj] = 1  
                        else:
                            adj[i * h + j, ii * h + jj] = kernel.flatten()[kernel_counter]
                    kernel_counter += 1
    return adj



def plot_adj_matrix(adj):
    G = nx.from_numpy_array(adj.numpy())
    nx.draw(G)
    plt.show()


def kernel_to_adj(adj, kernel):
    coeff = adj.clone()

    h, w = np.sqrt(adj.shape[0]).astype(int), np.sqrt(adj.shape[0]).astype(int)
    
    for i in range(w):
        for j in range(h):
            kernel_counter = 0
            for ii in range(i-1, i+2):
                for jj in range(j-1, j+2):
                    if 0 <= ii < w and 0 <= jj < h:
                        # coeff[i * h + j, ii * h + jj] = conv_kernel.flatten()[kernel_counter]
                        coeff[i * h + j, ii * h + jj] = kernel.flatten()[kernel_counter]
                    kernel_counter += 1

    return coeff



HANDLED_FUNCTIONS = {}
def implements(torch_function):
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func.__name__
        return func
    return decorator