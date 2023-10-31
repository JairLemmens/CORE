
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer_Encoder_Layer(nn.Module):
    
    def __init__(self,dim,num_heads=1 ,feed_forward_dim = None):
        super().__init__()

        if feed_forward_dim == None:
            feed_forward_dim = dim*4
    
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        #MLP
        self.linear1 = nn.Linear(dim,feed_forward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim,dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self,x):
        x2 = self.self_attn(x,x,x)[0]
        x = x + self.norm1(x2)
        
        #FeedForward
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x = x + x2
        x = self.norm2(x)
        
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

        #MLP
        self.linear1 = nn.Linear(dim,feed_forward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim,dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self,x,memory):
        x2 = self.self_attn(x,x,x)[0]
        x = x + self.norm1(x2)
        x2 , attn = self.multihead_attn(x,memory,memory)
        x = x + self.norm2(x2)
        
        #FeedForward
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x = x + x2
        x = self.norm3(x)
        
        return(x,attn)
    
    
class ATM_ViT(nn.Module):
    
    def __init__(self,dim,num_heads=8, n_classes = 2,qkv_bias=False, num_patches= 64, num_layers = 12, feed_forward_dim = None):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.encoder_layers.append(Transformer_Encoder_Layer(dim,num_heads,feed_forward_dim))
            self.decoder_layers.append(Transformer_Decoder_Layer(dim,num_heads,qkv_bias,feed_forward_dim))

        self.q = nn.Embedding(n_classes,dim)
        self.pos_emb = nn.Embedding(num_patches,dim)
        self.classifier = nn.Sequential(nn.Linear(n_classes,n_classes+1),nn.Softmax(-1))

    def forward(self,x):
        memory = []
        attns  = []

        #pos embedding
        x = x + self.pos_emb.weight

        #encoder
        for enc_layer in self.encoder_layers:
            memory.append(enc_layer(x))

        memory.reverse()

        #decoder
        batch_size = x.shape[0]
        q = self.q.weight.repeat(batch_size,1,1)

        for memory_item,dec_layer in zip(memory,self.decoder_layers):
            q, attn = dec_layer(q,memory_item)
            attns.append(attn)
        
        class_prediction = self.classifier(q.permute(0,2,1))

        return(q,attns,class_prediction)
        return(q,attns,class_prediction)