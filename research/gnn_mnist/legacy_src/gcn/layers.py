import torch
import torch.nn as nn

from utils import image_to_adj



class GCLayer(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, bias) -> None:
        super(GCLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels, 1, 1, 1), requires_grad=True))
        else:
            self.bias = None
        
        self.kernel = nn.Parameter(torch.rand(out_channels, 3, 3, requires_grad=True))

        self.register_buffer('adj', image_to_adj(torch.zeros((image_size, image_size))))

        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, image_size**2, image_size**2, requires_grad=False), requires_grad=True)
        # self.weight.data = self.weight.data + torch.eye(image_size**2).unsqueeze(0).unsqueeze(0)

        self._apply_constraint()
        self._mask_weights()


    def forward(self, x):
        self._mask_weights()
        self._apply_constraint()
        # self.weight = self.weight * self.adj
        
        dims = x.size()        
        x = x.view(x.size(0), x.size(1), -1, 1)
        out = []

        # apply batch matrix multiplication, can be optimized
        for i in range(x.size(0)):            
            out.append(torch.matmul(self.weight, x[i]))
                
        x = torch.stack(out)


        if self.bias is not None: # bias is the last thing to be added, but we are adding it here. to be checked later
            x = x + self.bias
        
        x = x.sum(dim=2) # are we sure about this? (it is sum for torch.nn.convnd)
        x = x.view(dims[0], self.out_channels, dims[2], dims[3])
        return x


    def _mask_weights(self):
        with torch.no_grad():
            # self.weight = nn.Parameter(self.weight * self.adj)
            self.weight.data = self.weight * self.adj


    def _apply_constraint(self):
        with torch.no_grad():

            w = self.image_size

            for c in range(self.out_channels):
                for i in range(w):
                    for j in range(w):
                        kernel_counter = 0
                        for ii in range(i-1, i+2):
                            for jj in range(j-1, j+2):
                                if 0 <= ii < w and 0 <= jj < w:
                                    # coeff[i * h + j, ii * h + jj] = conv_kernel.flatten()[kernel_counter]
                                    self.weight[c, :, i * w + j, ii * w + jj] = self.kernel[c].flatten()[kernel_counter]
                                    
                                kernel_counter += 1

            # for i in range(self.image_size**2):
            #     for j in range(self.d.size//2):
            #         if 0 <= i + self.d[j] < self.image_size**2:
            #             self.weight[:, :, i, i+j] = self.weight[:, :, i, i-j]
            
            # for i in range(self.image_size**2):
            
            #     if i < self.image_size**2 - 1:
            #         self.weight[:, :, i, i+1] = self.weight[:, :, i+1, i]
                
            #     if i < self.image_size**2 - self.image_size:
            #         self.weight[:, :, i, i+self.image_size] = self.weight[:, :, i+self.image_size, i]
            #     if i < self.image_size**2 - self.image_size - 1:
            #         self.weight[:, :, i, i+self.image_size+1] = self.weight[:, :, i+self.image_size+1, i] 
            #     if i < self.image_size**2 - self.image_size + 1:
            #         self.weight[:, :, i, i+self.image_size-1] = self.weight[:, :, i+self.image_size-1, i]



class BlockGC(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, bias) -> None:
        super(BlockGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels, 1, 1, 1), requires_grad=True))
        else:
            self.bias = None
        
        self.kernel = nn.Parameter(torch.rand(out_channels, 3, 3, requires_grad=True))

        self.register_buffer('adj', image_to_adj(torch.zeros((image_size, image_size))))

        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, image_size**2, image_size**2, requires_grad=False), requires_grad=True)
        # self.weight.data = self.weight.data + torch.eye(image_size**2).unsqueeze(0).unsqueeze(0)

        self._mask_weights()
        self._apply_constraint()


    def forward(self, x):
        self._mask_weights()
        self._apply_constraint()
        # self.weight = self.weight * self.adj
        
        dims = x.size()        
        x = x.view(x.size(0), x.size(1), -1, 1)
        out = []

        # apply batch matrix multiplication
        for i in range(x.size(0)):            
            out.append(torch.matmul(self.weight, x[i].unsqueeze(0)))
                
        x = torch.stack(out)

        if self.bias is not None:
            x = x + self.bias
        
        x = x.mean(dim=2) # are we sure about this?
        x = x.view(dims[0], self.out_channels, dims[2], dims[3])
        return x


    def _mask_weights(self):
        with torch.no_grad():
            # self.weight = nn.Parameter(self.weight * self.adj)
            self.weight.data = self.weight * self.adj


    def _apply_constraint(self):
        with torch.no_grad():

            w = self.image_size

            for c in range(self.out_channels):
                for i in range(w):
                    for j in range(w):
                        kernel_counter = 0
                        for ii in range(i-1, i+2):
                            for jj in range(j-1, j+2):
                                if 0 <= ii < w and 0 <= jj < w:
                                    # coeff[i * h + j, ii * h + jj] = conv_kernel.flatten()[kernel_counter]
                                    self.weight[c, :, i * w + j, ii * w + jj] = self.kernel[c].flatten()[kernel_counter]
                                    
                                kernel_counter += 1