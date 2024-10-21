import types
import torch
import torch.nn as nn
from utils import implements, image_to_adj



# this class and the next class implementation assume the shape of the input is at most (out_channels, in_channels, kernel_size, kernel_size) not any more :)
class SemiRegSpraseBCSR(object):
    """
    Semi-Regular Sparse Block Compressed Sparse Row matrix
    TODO: 
        - for a general case of matmul, nonzero should be implemented/overridden
    """
    def __init__(self, shape, ptr, indices, data, blocks):
        # assert shape[0] == shape[1], "Only square matrices are supported"
        self._shape = shape # block matrix shape

        self.ptr = ptr
        self.indices = indices
        self.data = data # stores indices of the block patterns

        self.blocks = blocks
        self.block_size = blocks.shape[-1]
        self.shape = torch.Size(list(self.blocks.shape[:-3]) + [self.block_size*self._shape[0], self.block_size*self._shape[1]]) # dense matrix shape
        self.device = blocks.device


    def to_dense(self): # TODO: imeplement the general case (not all the stacked matrices are alike)
        res = torch.zeros(self.shape)
        for i in range(len(self.ptr)-1):
            for j in range(self.ptr[i], self.ptr[i+1]):
                idx = self.indices[j]
                res[..., i*self.block_size:(i+1)*self.block_size, idx*self.block_size:(idx+1)*self.block_size] = self.blocks[..., self.data[j], :, :]
        return res


    def _get_block(self, i, j):
        assert i < self._shape[0] and j < self._shape[1], "Index out of bounds"
        if j in self.indices[self.ptr[i]: self.ptr[i+1]]:
            idx = self.indices[self.ptr[i]: self.ptr[i+1]].index(j) + self.ptr[i]
            return self.blocks[..., self.data[idx], :, :]
    
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
        return torch.stack([self._get_block(i, j) for j in self.indices[self.ptr[i]:self.ptr[i+1]]], dim=-3) # -3 is the axis of the blocks


    def __getitem__(self, key):
        # Note: Assuming every index is in normal matrix indexing space
        if not isinstance(key, tuple):
            key = (key,)

        assert len(key) <= len(self.shape), "Invalid number of indices"

        ellipsis_c = 0
        i = 0
        while i < len(key):
            if isinstance(key[i], types.EllipsisType):
                # add none slices for all the dimensions that are not indexed
                # find the last and next non ellipsis index
                ellipsis_c += 1
                assert ellipsis_c <= 1, "Only one ellipsis is supported"
                last = 0
                next = len(key)
                for j in range(i+1, len(key)):
                    if key[j] != types.EllipsisType:
                        next = j
                        break
                changed = False
                for j in range(i-1, -1, -1):
                    if key[j] != types.EllipsisType:
                        last = j
                        changed = True
                        break
                key = key[:last+changed] + tuple([slice(None)]*(len(self.shape) - len(key) + next-last-changed)) + key[next:]
            if isinstance(key[i], int):
                # print("getitem ", key[i], self.shape[i])
                assert key[i] < self.shape[i] and key[i]+self.shape[i] >= 0, "Index out of bounds"
                if key[i] < 0:
                    key = key[:i] + (self.shape[i]+key[i],) + key[i+1:]
                key = key[:i] + (key[i],) + key[i+1:]
            elif isinstance(key[i], slice):
                start = key[i].start if key[i].start is not None else 0
                stop = key[i].stop if key[i].stop is not None else self.shape[i]
                key = key[:i] + (slice(start, stop),) + key[i+1:]
                # print("getitem ", start, stop, self.shape)
                assert start >= 0 and start < self.shape[i] and stop > 0 and stop <= self.shape[i], "Index out of bounds"
            i += 1


        for i in range(len(key)): # checking the bounds
            if isinstance(key[i], slice):
                start = key[i].start if key[i].start is not None else 0
                stop = key[i].stop if key[i].stop is not None else self.shape[i]
                key = key[:i] + (slice(start, stop),) + key[i+1:]
                assert start >= 0 and start < self.shape[i] and stop > 0 and stop <= self.shape[i], "Index out of bounds"

        if len(self.shape) - len(key) >= 2: # batch slicing
            return SemiRegSpraseBCSR(self._shape, self.ptr, self.indices, self.data, self.blocks[key].squeeze())
        elif len(self.shape) - len(key) < 2: # block slicing
            related = len(self.shape) - 2
            # assert that the block shape wont be changed
            assert key[related].start % self.block_size == 0 and key[related].stop % self.block_size == 0, "Slicing is only supported on block boundaries"
            
            ptr = self.ptr[key[related].start//self.block_size:(key[related].stop//self.block_size)+1]
            indices = self.indices[ptr[0]:ptr[-1]]
            data = self.data[ptr[0]:ptr[-1]]
            ptr = [0] + [p-ptr[0] for p in ptr[1:]]
            
            if len(key) - related == 1: # row slicing
                shape = ((key[related].stop-key[related].start) // self.block_size, self._shape[1])
                new_ptr = ptr
                new_indices = indices
                new_data = data
            else: # row and column slicing
                assert key[related+1].start % self.block_size == 0 and key[related+1].stop % self.block_size == 0, "Slicing is only supported on block boundaries"
                shape = ((key[related].stop-key[related].start) // self.block_size, (key[related+1].stop-key[related+1].start) // self.block_size)
                
                # slice the columns
                new_indices = []
                new_data = []
                new_ptr = [0]
                for idx in range(len(ptr)-1):
                    col_indices = indices[ptr[idx]:ptr[idx+1]]
                    col_data = data[ptr[idx]:ptr[idx+1]]

                    temp_indices = []
                    temp_data = []
                    
                    start = key[related+1].start // self.block_size
                    stop = key[related+1].stop // self.block_size
                    
                    for i in range(len(col_indices)):
                        if col_indices[i] >= start and col_indices[i] < stop:
                            temp_indices.append(col_indices[i] - start)
                            temp_data.append(col_data[i])
                    
                    new_indices.extend(temp_indices)
                    new_ptr.append(len(temp_indices)+new_ptr[-1])
                    new_data.extend(temp_data)

            return SemiRegSpraseBCSR(shape, new_ptr, new_indices, new_data, self.blocks[key[:related]])            


    def __repr__(self) -> str:
        return f"Sparse Block Compressed Sparse Row matrix of shape {self._shape} and block size {self.block_size}"


    @implements(torch.stack)
    def stack(tensors, dim: int, device='cuda'):
        assert dim==0, "Only dim=0 is supported"
        print("here in stack")
        shape = tensors[0]._shape
        ptr = tensors[0].ptr
        indices = tensors[0].indices
        data = tensors[0].data
        blocks = torch.stack([t.blocks for t in tensors], dim=-4)
        return SemiRegSpraseBCSR(shape, ptr, indices, data, blocks)

    def matmul(self, other):
        assert self.shape[-2] == other.shape[-1], "Matrix multiplication is only possible with compatible shapes"
        res = torch.zeros((*self.shape[:-2], other.shape[-1], other.shape[-1]), device=self.device)

        for s in range(self.shape[-2]//self.block_size): # iterate over the rows of self
            idxs = self.indices[self.ptr[s]:self.ptr[s+1]]
            a = self.get_row(s)
            for o in range(other.shape[-1]//self.block_size): # iterate over the columns of other
                b = other[..., o*self.block_size:(o+1)*self.block_size]
                b = torch.stack([b[..., id*self.block_size:(id+1)*self.block_size, :].to_dense() for id in idxs], dim=-3)
                temp = torch.matmul(a, b)
                res[..., s*self.block_size:(s+1)*self.block_size, o*self.block_size:(o+1)*self.block_size] = torch.sum(temp, dim=-3)

        return res


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        
        if func == torch.stack:
            return cls.stack(*args, **kwargs)
        

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