import torch
import torch.nn as nn
import torch.nn.functional as F


#from ConvNext paper
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x




class Conv_block(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.depth_conv = nn.Conv2d(dim,dim,kernel_size=7,padding=3,groups=dim)
        self.layer_norm = LayerNorm(dim, data_format='channels_last')
        self.conv_1 = nn.Linear(dim,dim*4)
        self.act = nn.GELU()
        self.conv_2 = nn.Linear(dim*4,dim)

    def forward(self,x):
        input = x
        x = self.depth_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = x.permute(0, 3, 1, 2)
        return(x+input)
    


class Encoder(nn.Module):
    def __init__(self,in_chans= 3, depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],device = 'cpu'):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        in_layer = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0],data_format='channels_first')
        )
        self.layers.append(in_layer)
      
        for layer_n,depth in enumerate(depths):
            for sublayer_n in range(depth):
                self.layers.append(Conv_block(dims[layer_n]))
            if layer_n < len(depths)-1:
                downsample_block =  nn.Sequential(LayerNorm(dims[layer_n],data_format='channels_first'),
                                    nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))
                self.layers.append(downsample_block)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)

    
class Decoder(nn.Module):
    def __init__(self,in_chans=1 ,out_chans=3 ,depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],device = 'cpu'):
        super().__init__()
        self.device = device
        self.depths = list(reversed(depths))
        self.dims = list(reversed(dims))
        self.layers = nn.ModuleList()
        for layer_n,depth in enumerate(self.depths):

            for sublayer_n in range(depth):
                self.layers.append(Conv_block(self.dims[layer_n]))
            if layer_n < len(depths)-1:
                upsample_block = nn.Sequential(LayerNorm(self.dims[layer_n],data_format='channels_first'),
                                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                            nn.Conv2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=7,padding=3,groups=self.dims[layer_n+1]))
                self.layers.append(upsample_block)
        
        out_layer = nn.Sequential(LayerNorm(self.dims[-1],data_format='channels_first'),
                nn.ConvTranspose2d(self.dims[-1], out_chans,kernel_size=4,stride=4))
        self.layers.append(out_layer)


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)