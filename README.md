# Block GNN
The regularity of grids can be exploited to calculate the convolution operation efficiently. 
Table below states the speedup of the block convolution operation over the regular convolution operation implemented in PyTorch.

| Grid size | Speedup |
|-----------|---------|
| 8x8       | 1.5     |
| 28x28     | 2.0     |

See [this notebook](/notebooks/cnn.ipynb) for more details.

# Getting started
Check out the cnn notebook to see how to calculate the convolution operation on a grid faster than the regular convolution operation implemented in PyTorch.

```bash
pip install -e .
```



# Block convolution
The block convolution operation is faster than the regular convolution operation implemented in PyTorch. The block convolution operation is implemented in the `block_gnn` module. The block convolution operation is implemented in the `BlockConv` class. The `BlockConv` class is a subclass of the `torch.nn.Module` class. The `BlockConv` class has the following signature:
```python
class BlockConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(BlockConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.block = Block(in_channels, out_channels, kernel_size)
```




# References
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)