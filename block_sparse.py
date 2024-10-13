import torch
import torch.nn as nn
from utils import implements, image_to_adj



# this class and the next class implementation assume the shape of the input is at most (out_channels, in_channels, kernel_size, kernel_size)
class SemiRegSpraseBCSR(object):
    """
    Semi-Regular Sparse Block Compressed Sparse Row matrix
    TODO: 
        - for a general case of matmul, nonzero should be implemented/overridden
    """
    def __init__(self, shape, ptr, indices, data, blocks):
        assert shape[0] == shape[1], "Only square matrices are supported"
        self._shape = shape # block matrix shape

        self.ptr = ptr
        self.indices = indices
        self.data = data # stores indices of the block patterns

        self.blocks = blocks
        self.block_size = blocks.shape[-1]
        self.shape = torch.Size(list(self.blocks.shape[:-3]) + [self.block_size**2, self.block_size**2]) # dense matrix shape


    def to_dense(self): # TODO: imeplement the general case (not all the stacked matrices are alike)
        res = torch.zeros(self.shape)
        for i in range(len(self.ptr)-1):
            for j in range(self.ptr[i], self.ptr[i+1]):
                idx = self.indices[j]
                res[..., i*self.block_size:(i+1)*self.block_size, idx*self.block_size:(idx+1)*self.block_size] = self.blocks[..., self.data[j], :, :]
        return res


    def _get_block(self, i, j, channels=None):
        assert i < self._shape[0] and j < self._shape[1], "Index out of bounds"
        if j in self.indices[self.ptr[i]: self.ptr[i+1]]:
            idx = self.indices[self.ptr[i]: self.ptr[i+1]].index(j) + self.ptr[i]
            if len(self.blocks.shape) == 3:
                return self.blocks[self.data[idx]]
            if len(self.blocks.shape) == 5:
                if channels is not None:
                    if len(channels) == 2:
                        return self.blocks[..., self.data[idx], :, :]
                    elif len(channels) == 1:
                        raise ValueError("Invalid number of channels")
                        return self.blocks[channels[0]][:][self.data[idx]]
                else:
                    raise ValueError("Invalid number of channels")
            elif len(self.blocks.shape) == 4:
                return self.blocks[:, self.data[idx], :, :]
    
        else:
            raise ValueError("Accessing a zero (null) block")
            
    
    def set_block(self, i, j, block):
        assert i < self._shape[0] and j < self._shape[1], "Index out of bounds"
        if j in self.indices[self.ptr[i]: self.ptr[i+1]]:
            idx = self.indices[self.ptr[i]: self.ptr[i+1]].index(j) + self.ptr[i]
            self.blocks[self.data[idx]] = block
        else:
            raise ValueError("Accessing a zero (null) block")
        

    def get_row(self, i):
        return torch.stack([self._get_block(i, j, [0, 1]) for j in self.indices[self.ptr[i]:self.ptr[i+1]]], dim=-3) # -3 is the axis of the blocks


    def __getitem__(self, key):
        # get block is not supported here TODO: implement it!
        
        if not isinstance(key, tuple):
            key = (key,)

        if len(self.blocks.shape) == 3:
            if isinstance(key, int):
                try:
                    return SemiRegSpraseBCSR(self._shape, self.ptr, self.indices, self.data, self.blocks.unsqueeze(0)[key])
                except:
                    raise ValueError("Invalid index")
            else:
                raise ValueError("Invalid index, cannot slice a 2d block matrix")
        elif len(self.blocks.shape) > 3:
            assert len(self.blocks.shape) - len(key) >= 3, "Invalid number of indices"
            return SemiRegSpraseBCSR(self._shape, self.ptr, self.indices, self.data, self.blocks[key])
        else:
            raise ValueError("Invalid index, meh")
        

    def __repr__(self) -> str:
        return f"Sparse Block Compressed Sparse Row matrix of shape {self._shape} and block size {self.block_size}"


HANDLED_FUNCTIONS = {
    torch.matmul: "matmul",
    torch.stack: "stack"
}
class SparseAdjacency(SemiRegSpraseBCSR):
    def __init__(self, shape, ptr=None, indices=None, data=None, blocks=None, kernel_size=3, device='cuda'):
        assert kernel_size % 2 == 1, "Only odd kernel sizes are supported"
        
        self.kernel_size = kernel_size
        self.device = device
        
        if ptr is None:
            ptr = []
            indices = []
            data = []
            blocks = []
            
            for _ in range(kernel_size):
                connectivity = torch.zeros(shape[0], shape[1])
                for i in range(-kernel_size//2+1, kernel_size//2+1):
                    connectivity += torch.diag(torch.ones(shape[0]-abs(i)), i)

                blocks.append(connectivity)

            for i in range(shape[0]):
                ptr.append(len(indices))
                if i < kernel_size//2:
                    temp = [j+i for j in range(-i, kernel_size//2+1)]                    
                    data.extend([j+kernel_size//2 for j in range(-i, kernel_size//2+1)])
                elif i >= shape[0]-kernel_size//2:
                    temp = [j+i for j in range(-kernel_size//2+1, shape[0]-i)]                    
                    data.extend([j+kernel_size//2 for j in range(-kernel_size//2+1, shape[0]-i)])
                else:
                    temp = [j+i for j in range(-kernel_size//2+1, kernel_size//2+1)]                    
                    data.extend([j+kernel_size//2 for j in range(-kernel_size//2+1, kernel_size//2+1)])
                indices.extend(temp)
            ptr.append(len(indices))
            blocks = torch.stack(blocks, dim=0)
            blocks = blocks.to(device)
        super(SparseAdjacency, self).__init__(shape, ptr, indices, data, blocks)


    @implements(torch.stack)
    def stack(tensors, dim: int, device='cuda'):
        assert dim==0, "Only dim=0 is supported"
        shape = tensors[0]._shape
        ptr = tensors[0].ptr
        indices = tensors[0].indices
        data = tensors[0].data
        kernel_size = tensors[0].kernel_size
        blocks = [t.blocks for t in tensors]
        blocks = torch.stack(blocks, dim=0)
        return SparseAdjacency(shape, ptr, indices, data, blocks, kernel_size)
    
    
    def matmul(self, other): # HEADS UP: this function does not support the general case of matmul
        # TODO: this assertion should be extended to support multiple dimensions
        # assert self.shape[-1] == other.shape[-2], "Matrix multiplication is only possible with compatible shapes"

        res = torch.zeros((self.shape[0], self.shape[1], other.shape[-1], other.shape[-1]), device=self.device)
                            # out_c         in_c        image_size        image_size
        for i in range(self.block_size):
            if i < self.kernel_size//2:
                temp = torch.matmul(self[:, :].get_row(i), other[:, :i+self.kernel_size//2+1].unsqueeze(-1))
            elif i >= self.block_size - self.kernel_size//2:
                temp = torch.matmul(self[:, :].get_row(i), other[:, i-self.kernel_size//2:].unsqueeze(-1))
            else:
                temp = torch.matmul(self[:, :].get_row(i), other[:, i-self.kernel_size//2:i+self.kernel_size//2+1].unsqueeze(-1))
            res[:, :, i] = torch.sum(temp, dim=2).squeeze(-1)
        
        return res


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        
        if func == torch.stack:
            return cls.stack(*args, **kwargs)
        elif func == torch.matmul:
            return cls.matmul(*args, **kwargs)
        

class BlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, bias, device='cuda') -> None:
        super(BlockConv, self).__init__()
        
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.image_size = image_size

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1, 1), requires_grad=True)
            self.bias.data = self.bias.to(device)
        else:
            self.bias = None

        self.register_buffer('adj', image_to_adj(torch.zeros((image_size, image_size)), torch.ones((kernel_size, kernel_size)))[:image_size, :image_size])
        self.adj = self.adj.to(device)
        
        self.kernel = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True))
        self.kernel.data = self.kernel.to(device)

        self.coeff  = torch.stack([SparseAdjacency((image_size, image_size), kernel_size=kernel_size, device=device) for i in range(self.in_channels)], dim=0)
        self.coeff  = torch.stack([self.coeff for i in range(self.out_channels)], dim=0)

        self._mask_weights()
        self._apply_constraint()

    def forward(self, x):

        self._mask_weights()
        self._apply_constraint()

        dims = x.size()
        out = []
        
        res = torch.zeros_like(x)
        for b in range(x.size(0)): # batch operation, can be optimized
            out.append(self.coeff.matmul(x[b]))
        
        x = torch.stack(out)
        if self.bias is not None:
            x = res + self.bias
        x = x.sum(dim=2)
        
        return x
    

    def _mask_weights(self):
        with torch.no_grad():
            for block in self.coeff.blocks:
                block *= self.adj

    def _apply_constraint(self):
        with torch.no_grad():
            # for e, block in enumerate(self.coeff.blocks):
            for k in range(self.kernel_size):
                for i in range(self.image_size):
                    # for j in range(-self.kernel_size//2+1, self.kernel_size//2+1):
                    if i < self.kernel_size//2:                        
                        self.coeff.blocks[:, :, k, i, :i+self.kernel_size//2+1] = self.kernel[:, :, k, -i+self.kernel_size//2:]
                    elif i >= self.image_size - self.kernel_size//2:                        
                        self.coeff.blocks[:, :, k, i, i-self.kernel_size//2:] = self.kernel[:, :, k, :self.image_size-i+self.kernel_size//2]
                    else:                        
                        self.coeff.blocks[:, :, k, i, i-self.kernel_size//2:i+self.kernel_size//2+1] = self.kernel[:, :, k]