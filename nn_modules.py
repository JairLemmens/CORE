import torch
import torch.nn as nn
import torch.nn.functional as F


#from ConvNext paper
# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError 
#         self.normalized_shape = (normalized_shape, )
    
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x



# With layernorm
# class Conv_block(nn.Module):
#     def __init__(self,dim,dConv_kernel_size=7):
#         super().__init__()
#         self.depth_conv = nn.Conv2d(dim,dim,kernel_size=dConv_kernel_size,padding=int((dConv_kernel_size-1)/2),groups=dim)
#         self.layer_norm = LayerNorm(dim, data_format='channels_last')
#         self.conv_1 = nn.Linear(dim,dim*4)
#         self.act = nn.GELU()
#         self.conv_2 = nn.Linear(dim*4,dim)

#     def forward(self,x):
#         input = x
#         x = self.depth_conv(x)
#         x = x.permute(0, 2, 3, 1)
#         x = self.layer_norm(x)
#         x = self.conv_1(x)
#         x = self.act(x)
#         x = self.conv_2(x)
#         x = x.permute(0, 3, 1, 2)
#         return(x+input)



class Conv_block(nn.Module):
    def __init__(self,dim,dConv_kernel_size=7):
        super().__init__()
        self.depth_conv = nn.Conv2d(dim,dim,kernel_size=dConv_kernel_size,padding=int((dConv_kernel_size-1)/2),groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.conv_1 = nn.Conv2d(dim,dim*4,kernel_size=1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(dim*4,dim,kernel_size=1)

    def forward(self,x):
        input = x
        x = self.depth_conv(x)
        x = self.norm(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        return(x+input)
    

#pre 6/10
# class Encoder(nn.Module):
#     def __init__(self,in_chans= 3, depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536]):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         in_layer = nn.Sequential(
#                 nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#                 LayerNorm(dims[0],data_format='channels_first')
#         )
#         self.layers.append(in_layer)
      
#         for layer_n,depth in enumerate(depths):
#             for sublayer_n in range(depth):
#                 self.layers.append(Conv_block(dims[layer_n]))
#             if layer_n < len(depths)-1:
#                 downsample_block =  nn.Sequential(LayerNorm(dims[layer_n],data_format='channels_first'),
#                                     nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))
#                 self.layers.append(downsample_block)

#     def forward(self,x):
#         for layer in self.layers:
#             x = layer(x)
#         return(x)

#pre 6/10  
# class Decoder(nn.Module):
#     def __init__(self,in_chans=1 ,out_chans=3 ,depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536]):
#         super().__init__()
#         self.depths = list(reversed(depths))
#         self.dims = list(reversed(dims))
#         self.layers = nn.ModuleList()
#         for layer_n,depth in enumerate(self.depths):

#             for sublayer_n in range(depth):
#                 self.layers.append(Conv_block(self.dims[layer_n]))
#             if layer_n < len(depths)-1:
#                 upsample_block = nn.Sequential(LayerNorm(self.dims[layer_n],data_format='channels_first'),
#                                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#                                             nn.Conv2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=7,padding=3,groups=self.dims[layer_n+1]))
#                 self.layers.append(upsample_block)
        
#         out_layer = nn.Sequential(LayerNorm(self.dims[-1],data_format='channels_first'),
#                 nn.ConvTranspose2d(self.dims[-1], out_chans,kernel_size=4,stride=4))
#         self.layers.append(out_layer)


#     def forward(self,x):
#         for layer in self.layers:
#             x = layer(x)
#         return(x)



#with LayerNorm 06/10
# class Encoder(nn.Module):
#     def __init__(self,in_chans= 3, depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
#         super().__init__()
#         self.layers = nn.ModuleList()
      
#         for layer_n,depth in enumerate(depths):
#             for sublayer_n in range(depth):
#                 self.layers.append(Conv_block(dims[layer_n],dConv_kernel_size))
#             if layer_n < len(depths)-1:
#                 downsample_block =  nn.Sequential(LayerNorm(dims[layer_n],data_format='channels_first'),
#                                     nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))
#                 self.layers.append(downsample_block)

#     def forward(self,x):
#         for layer in self.layers:
#             x = layer(x)
#         return(x)
    

# class Decoder(nn.Module):
#     def __init__(self,in_chans=1 ,out_chans=3 ,depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
#         super().__init__()
#         self.depths = list(reversed(depths))
#         self.dims = list(reversed(dims))
#         self.layers = nn.ModuleList()
#         for layer_n,depth in enumerate(self.depths):

#             for _ in range(depth):
#                 self.layers.append(Conv_block(self.dims[layer_n],dConv_kernel_size))
#             if layer_n < len(depths)-1:
#                 upsample_block = nn.Sequential(LayerNorm(self.dims[layer_n],data_format='channels_first'),
#                                             nn.ConvTranspose2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=2,stride=2))
#                 self.layers.append(upsample_block)

#     def forward(self,x):
#         for layer in self.layers:
#             x = layer(x)
#         return(x)

#without layernorm 06/10
class Encoder(nn.Module):
    def __init__(self,in_chans= 3, depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
        super().__init__()
        self.layers = nn.ModuleList()
      
        for layer_n,depth in enumerate(depths):
            for sublayer_n in range(depth):
                self.layers.append(Conv_block(dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:
                self.layers.append(nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)
    

class Decoder(nn.Module):
    def __init__(self,in_chans=1 ,out_chans=3 ,depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
        super().__init__()
        self.depths = list(reversed(depths))
        self.dims = list(reversed(dims))
        self.layers = nn.ModuleList()
        for layer_n,depth in enumerate(self.depths):

            for _ in range(depth):
                self.layers.append(Conv_block(self.dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:     
                self.layers.append(nn.ConvTranspose2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=2,stride=2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)



class ATM(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias =False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim// num_heads
        self.scale = qk_scale or head_dim ** -5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)

    def forward(self,xq,xk,xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]
        
        q = self.q(xq).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        return x, attn_save.sum(dim=1) / self.num_heads
    



class Transformer_Decoder_Layer(nn.Module):
    
    def __init__(self,dim,num_heads=1,qkv_bias=False,feed_forward_dim = None):
        super().__init__()

        if feed_forward_dim == None:
            feed_forward_dim = dim*4
    
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.multihead_attn =ATM(dim,num_heads,qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim,feed_forward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim,dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self,x,memory):
        x2 = self.self_attn(x,x,x)[0]
        x += self.norm1(x2)

        #if we add a Transformer Encoder instead of the convolutional one we can use the attn to make a nice graphic of the attention map but right now it is spatially meaningless
        x2 , attn = self.multihead_attn(x,memory,memory)
        x += self.norm2(x2)
        
        #FeedForward
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x+= x2
        x = self.norm3(x)
        
        return(x,attn)