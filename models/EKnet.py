from ast import Not
from functools import partial
from pickle import NONE, TRUE
from tkinter import N
from einops import repeat, rearrange
import numpy as np
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    from timm
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_2d_sincos_pos_embed(embed_dim, grid_size_w=16, grid_size_h=12, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    # print('grid',grid.shape)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # print('pos_embed',pos_embed.shape)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadAttention_orginal(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key=None, value=None, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# from https://github.com/hila-chefer/Transformer-Explainability
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.content_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        # self.attention_map = None
        self.attention_map = {}

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        # return self.attention_map
        return self.attention_map

    # def forward(self, x,tgt, memory, register_hook=False):
    def forward(self, query, key=None, value=None, register_hook=False, return_atten=False):
        h = self.num_heads
        b, n_q , _ = query.shape
        _, n_k, _ = key.shape

        query = self.query_proj(query)
        key = self.content_proj(key)
        value = self.value_proj(value)

        q = rearrange(query, 'b n_q (h d) -> b h n_q d', h = h)
        k = rearrange(key, 'b n_k (h d) -> b h n_k d', h = h)
        v = rearrange(value, 'b n_k (h d) -> b h n_k d', h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        if register_hook:
            # self.save_attention_map(attn)
            self.attention_map[query.device] = attn
            # print('self.attention_map',self.attention_map[query.device].shape)
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.proj(out)
        out = self.proj_drop(out)
        return out





class selfAttention_layer(nn.Module):
    '''
    vit 方向 -> x+pos 提前加好， 等于 q,k,v 都加
    bert 方向 -> x和pos 里面加, q+pos, k+pos, v ,  等于 q,k 加, v不加
    这里就用vit的方向
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                normalize_before=True,
                proj_drop=0., attn_drop=0., drop_path=0.0, 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        self.normalize_before = normalize_before

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x,register_hook=False):
        x1_norm = self.norm1(x)
        x = x + self.drop_path(self.attn(x1_norm,key=x1_norm,value=x1_norm,register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    '''orginal'''
    # def forward_post(self, x, register_hook=False):
    #     x = self.norm1(x + self.drop_path(self.attn(x,key=x,value=x,register_hook=register_hook)))
    #     x = self.norm2(x + self.drop_path(self.mlp(x)))
    #     return x

    def forward_post(self, x, register_hook=False):
        x = x + self.drop_path(self.norm1(self.attn(x,key=x,value=x,register_hook=register_hook)))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


    def forward(self, x,register_hook=False):
        '''
        '''
        if self.normalize_before:
            return self.forward_pre(x,register_hook=register_hook)
        return self.forward_post(x,register_hook=register_hook)

class crossAttention_layer(nn.Module):
    '''
    vit 方向 -> x+pos 提前加好， 等于 q,k,v 都加
    bert 方向 -> x和pos 里面加, q+pos, k+pos, v ,  等于 q,k 加, v不加
    这里就用vit的方向
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                normalize_before=True,
                proj_drop=0., attn_drop=0., drop_path=0.0, 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        self.normalize_before = normalize_before

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x ,key=None, value=None,register_hook=False):
        x1_norm = self.norm1(x)
        x = x + self.drop_path(self.attn(x1_norm,key=key,value=value,register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    '''orginal'''
    # def forward_post(self, x, key=None, value=None,register_hook=False):
    #     x = self.norm1(x + self.drop_path(self.attn(x,key=key,value=value,register_hook=register_hook)))
    #     x = self.norm2(x + self.drop_path(self.mlp(x)))
    #     return x

    '''sw v2'''
    def forward_post(self, x, key=None, value=None,register_hook=False):
        x = x + self.drop_path(self.norm1(self.attn(x,key=key,value=value,register_hook=register_hook)))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


    def forward(self, x, key=None,value=None,register_hook=False):
        '''
        '''
        if self.normalize_before:
            return self.forward_pre(x,key=key, value=value,register_hook=register_hook)
        return self.forward_post(x,key=key, value=value,register_hook=register_hook)

class PatchEmbed_2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=250, embed_dim=768):
        super().__init__()
        self.projs = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), stride=(1,patch_size))

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.unsqueeze(1) # 2,12,5000 BCW -> 2,1,12,5000 BCHW
        x = self.projs(x) # 2, 512, 12, 10
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC # 2, 512, 12, 10  -> # 2, 512, 120
        return x

class PatchEmbed_2D_time_all(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, input_c=3, patch_size=250, embed_dim=768):
        super().__init__()
        self.projs = nn.Conv2d(1, embed_dim, kernel_size=(input_c,patch_size), stride=(1,patch_size))

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.unsqueeze(1) # 2,12,5000 BCW -> 2,1,12,5000 BCHW
        # print('x unsqueeze',x.shape) 
        x = self.projs(x) # -> bs,dim,1, n_patch  3,32,1,200
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC # 2, 512, 12, 10  -> # 2, 512, 120
        return x

class PatchEmbed_1D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, input_c,patch_size=250, embed_dim=768):
        super().__init__()
        # self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.projs = nn.ModuleList([
                    nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
                    for i in range(input_c)])
        
    def forward(self, x):
        ## bs,c,1250
        xs = []
        for i in range(x.shape[1]):
            x_i = x[:,i,:]
            # print('x_i',x_i.shape)
            x_i = x_i.unsqueeze(1) # bs,1250 -> bs,1,1250
            # print('x_i',x_i.shape)
            x_i = self.projs[i](x_i)  # bs,1,1250 -> bs,embed_dim,1250/patch_size
            # print('x_i',x_i.shape)
            xs.append(x_i)
        xs = torch.cat(xs, 2) #  bs,embed_dim,1250/patch_size -> bs,embed_dim,(1250/patch_size)*c
        xs = xs.transpose(1, 2) #  bs,embed_dim,(1250/patch_size)*c -> bs,(1250/patch_size)*c,embed_dim
        # print('xs',xs.shape) # 2 512,120
        return xs


class MLP_Purification(nn.Module):
    def __init__(self, in_dim, keep_ratio=0.5,  act_layer=nn.GELU,):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim//4)
        self.act = act_layer()
        # self.fc2 = nn.Linear(in_dim//2, in_dim//4)
        self.fc3 = nn.Linear(in_dim//4, 1)
        self.keep_ratio = keep_ratio
        # self.fc_expand = nn.Linear(in_dim, out_dim)

    def forward(self, input, positin_embedding):
        N,L,D = input.shape
        len_keep = int(L*self.keep_ratio)
        # print('MLP_Purification in',input.shape)
        # print('positin_embedding',positin_embedding.shape)

        x = self.fc0(input)
        x = self.act(x)

        x = self.fc1(x)
        x = self.act(x)

        # x = self.fc2(x)
        # x = self.act(x)

        x = self.fc3(x) # bs,L,1 
        x = x.squeeze(-1) #bs,L,1 -> bs,L

        # print('MLP_Purification out',x,x.shape)
        x = x.softmax(dim=-1)
        # print('MLP_Purification softmax',x,x.shape)
        
        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]   [2, 120]
        ids_sorted = torch.argsort(x, dim=1,descending=True)  # large to small # bs,L
        # ids_restore = torch.argsort(ids_sorted, dim=1)

        # print('ids_sorted',ids_sorted,ids_sorted.shape) 
        ids_keep = ids_sorted[:, :len_keep]

        # print('input',input.shape)
        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print('x_ gather',x_selected.shape)
        
        # x_selected = self.fc_expand(x_selected)
        # print('x_selected expand',x_selected.shape)
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))
        # print('positin_embedding',positin_embedding.shape)
        # print('x_selected_pos',x_selected_pos.shape)
        # assert 1>2
        return x_selected,x_selected_pos


class Self_CrossKnowledge(nn.Module):
    def __init__(self, encoder_depth, embed_dim=32, 
                    num_heads=2,cls_cross_depth=3, norm_layer=nn.LayerNorm,
                    keep_ratio=0.5, normalize_before=False, 
                    mlp_ratio=4, act_layer=nn.GELU,):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            selfAttention_layer(embed_dim, num_heads, mlp_ratio, 
            normalize_before=normalize_before,
            qkv_bias=True,act_layer=act_layer)
            for i in range(encoder_depth)])


        self.encoder_norm = norm_layer(embed_dim)
        self.keep_ratio = keep_ratio
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        self.knowledge_self_layers = nn.ModuleList([
            selfAttention_layer(embed_dim, num_heads, mlp_ratio, 
            normalize_before=normalize_before,
            qkv_bias=True, act_layer=act_layer)
            for i in range(cls_cross_depth)])

        self.knowledge_cross_layers = nn.ModuleList([
            crossAttention_layer(embed_dim, num_heads, mlp_ratio, 
                normalize_before=normalize_before,
                qkv_bias=True,act_layer=act_layer)
                for i in range(cls_cross_depth)])


    def forward(self, input, knowledge_emb, pos_emb_i=None, register_hook=False):
        N,L,D = input.shape

        if pos_emb_i is not None:
            x = input+pos_emb_i
        else:
            x = input

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, register_hook=register_hook)
        x = self.decoder_embed(x)

        for self_atten_layer,cross_atten_layer in zip(self.knowledge_self_layers, self.knowledge_cross_layers):
            knowledge_emb = self_atten_layer(knowledge_emb, register_hook=register_hook) # 15,64
            knowledge_emb = cross_atten_layer(knowledge_emb, key=x, value=x, register_hook=register_hook) ## 15,64
        return knowledge_emb


class local_purificaiton_module(nn.Module):
    def __init__(self, encoder_depth, embed_dim=128,
                    num_heads=2,norm_layer=nn.LayerNorm,
                    keep_ratio=0.5, normalize_before=False, 
                    mlp_ratio=4, act_layer=nn.GELU,
                    n_leads=15,
                    pufication_module='softmax'):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            selfAttention_layer(embed_dim, num_heads, mlp_ratio, 
            normalize_before=normalize_before,
            qkv_bias=True,act_layer=act_layer)
            for i in range(encoder_depth)])

        if pufication_module == 'softmax':
            self.token_purification = MLP_Purification(embed_dim,keep_ratio=keep_ratio,act_layer=act_layer)
        elif pufication_module == 'token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        elif pufication_module == 'cross_emb':
            self.score_emb = nn.Embedding(1, embed_dim).requires_grad_(True)
        else:
            print(f'pufication_module {pufication_module} wrong')
        self.pufication_module = pufication_module

        self.encoder_norm = norm_layer(embed_dim)
        self.embed_dim = embed_dim
        self.keep_ratio = keep_ratio
        self.n_leads = n_leads

    def separately(self,x):
        self.ori_bs, self._total_p, self.d = x.shape
        x_local = x.reshape(self.ori_bs,self.n_leads,self._total_p//self.n_leads,self.d)
        # print('x reshape',x_local.shape)
        x_local = x_local.reshape(self.ori_bs*self.n_leads,self._total_p//self.n_leads,self.d)
        # print('x reshape',x_local.shape)
        return x_local

    def unseparately(self,x):
        _,total_p,d = x.shape # bs*lead,25,d
        x_local = x.reshape(self.ori_bs,self.n_leads,-1, d) # bs,15, 25,d
        # print('x reshape',x_local.shape)
        x_local = x_local.reshape(self.ori_bs,-1, self.d) #  bs,15*25,d
        # print('x reshape',x_local.shape)
        # assert 1>2
        return x_local

    def calc_atten_only(self,x):
        self.scale = self.embed_dim ** -0.5
        dots = torch.einsum('bid,bjd->bij', x, x) * self.scale
        attn = dots.softmax(dim=-1)
        return attn

    def sortANDgather(self,atten,input,positin_embedding):
        N,L,D = input.shape
        len_keep = int(L*self.keep_ratio)

        ids_sorted = torch.argsort(atten, dim=1,descending=True)  # large to small # bs,L

        ids_keep = ids_sorted[:, :len_keep]

        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))
        return x_selected,x_selected_pos

    def forward(self, x, pos_emb, register_hook=False):
        bs,total_p,d = x.shape

        x = x+pos_emb

        x_local = self.separately(x)
        pos_emb_local = self.separately(pos_emb)
        # print('x reshape',x_local.shape)
        # print('x pos_emb_local',pos_emb_local.shape)

        if self.pufication_module == 'token':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x_local.shape[0])
            x_local = torch.cat((cls_tokens, x_local), dim=1)

        for encoder_layer in self.encoder_layers:
            x_local = encoder_layer(x_local, register_hook=register_hook)

        if self.pufication_module == 'token':
            x_atten = self.calc_atten_only(x_local)
            x_atten_all = x_atten[:,0,1:]
            x_local = x_local[:,1:,:]

            x_local = self.encoder_norm(x_local)
            x_local_pured,pos_emb_local_pured = self.sortANDgather(x_atten_all,x_local,pos_emb_local)
        elif self.pufication_module == 'cross_emb':
            score_emb = self.score_emb.weight
            score_emb = score_emb.unsqueeze(0).repeat(x_local.shape[0],1, 1)

            self.scale = self.embed_dim ** -0.5
            dots = torch.einsum('bid,bjd->bij', score_emb, x_local) * self.scale
            x_atten = dots.softmax(dim=-1) # 150,1,50
            x_atten = x_atten.squeeze()
            # print('x_atten',x_atten.shape)
            
            x_local = self.encoder_norm(x_local)
            x_local_pured,pos_emb_local_pured = self.sortANDgather(x_atten,x_local,pos_emb_local)

        else:
            x_local = self.encoder_norm(x_local)
            x_local_pured,pos_emb_local_pured = self.token_purification(x_local,pos_emb_local)

        # print('x_local_pured',x_local_pured.shape)
        # print('pos_emb_local_pured',pos_emb_local_pured.shape)
        # assert 1>2
        x_pured = self.unseparately(x_local_pured)
        positin_embedding_pured = self.unseparately(pos_emb_local_pured)
        return x_pured,positin_embedding_pured




class MLP_Purification(nn.Module):
    def __init__(self, in_dim, pufication_ratio=0.5,  act_layer=nn.GELU,):
        '''

        input L,D
        out L/r, D
        '''
        super().__init__()
        self.fc0 = nn.Linear(in_dim, in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim//4)
        self.act = act_layer()
        # self.fc2 = nn.Linear(in_dim//2, in_dim//4)
        self.fc3 = nn.Linear(in_dim//4, 1)
        self.pufication_ratio = pufication_ratio
        # self.fc_expand = nn.Linear(in_dim, out_dim)

    def forward(self, input, positin_embedding, pure_ways='atten'):
        def shuffle(t):
            noise = torch.rand(t.shape[0], t.shape[1], device=input.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            t = torch.gather(t, dim=1, index=ids_shuffle)
            return t

        N,L,D = input.shape
        len_keep = int(L*self.pufication_ratio)
        
        if pure_ways not in ['atten','random','combine']:
            raise ValueError(f'{pure_ways} not support')
        
        x = self.fc0(input)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc3(x) # bs,L,1 
        x = x.squeeze(-1) #bs,L,1 -> bs,L

        x = x.softmax(dim=-1)

        # ids_sorted = torch.argsort(x, dim=1,descending=True)  # large to small # bs,L
        # ids_keep = ids_sorted[:, :len_keep]

        ids_sorted = torch.argsort(x, dim=1,descending=True)  # large to small # bs,L
        ids_keep_atten = ids_sorted[:, :len_keep]
        ids_keep_NOTatten = ids_sorted[:, len_keep:]

        noise = torch.rand(N, L, device=input.device)  # noise in [0, 1]   [2, 120]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_noise = ids_shuffle[:, :len_keep]

        if pure_ways in ['atten']:
            ids_keep=ids_keep_atten
        elif pure_ways in ['random']:
            ids_keep=ids_keep_noise
        elif pure_ways in ['combine']:
            ids_keep_atten = shuffle(ids_keep_atten)
            ids_keep_NOTatten = shuffle(ids_keep_NOTatten)
            ids_keep = torch.cat([ids_keep_atten[:,:len_keep//2],ids_keep_NOTatten[:,len_keep//2:len_keep]],dim=1)

        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))

        return x_selected,x_selected_pos


class Token_Purification(nn.Module):
    def __init__(self, in_dim, pufication_ratio=0.5):
        '''

        input L+1,D
        out L/r, D
        '''
        super().__init__()
        self.pufication_ratio = pufication_ratio
        # self.fc_expand = nn.Linear(in_dim, out_dim)
        self.embed_dim = in_dim


    def calc_atten_only(self,x):
        self.scale = self.embed_dim ** -0.5
        dots = torch.einsum('bid,bjd->bij', x, x) * self.scale
        attn = dots.softmax(dim=-1)
        return attn

    def sortANDgather(self,atten,input,positin_embedding,pure_ways='atten'):
        def shuffle(t):
            noise = torch.rand(t.shape[0], t.shape[1], device=input.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            t = torch.gather(t, dim=1, index=ids_shuffle)
            return t

        N,L,D = input.shape
        len_keep = int(L*self.pufication_ratio)


        ids_sorted = torch.argsort(atten, dim=1,descending=True)  # large to small # bs,L
        ids_keep_atten = ids_sorted[:, :len_keep]
        ids_keep_NOTatten = ids_sorted[:, len_keep:]

        noise = torch.rand(N, L, device=input.device)  # noise in [0, 1]   [2, 120]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_noise = ids_shuffle[:, :len_keep]

        if pure_ways in ['atten']:
            ids_keep=ids_keep_atten
        elif pure_ways in ['random']:
            ids_keep=ids_keep_noise
        elif pure_ways in ['combine']:
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_atten = shuffle(ids_keep_atten)
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_NOTatten = shuffle(ids_keep_NOTatten)
            ids_keep = torch.cat([ids_keep_atten[:,:len_keep//2],ids_keep_NOTatten[:,len_keep//2:len_keep]],dim=1)
        # print('ids_keep',ids_keep.shape)
        # assert 1>2
        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))

        return x_selected,x_selected_pos

    
    def forward(self, x_lead, x_pos_masked, pure_ways='atten'):
        if pure_ways not in ['atten','random','combine']:
            raise ValueError(f'{pure_ways} not support')
        
        x_atten = self.calc_atten_only(x_lead) # bs,n_patch+1,d -> bs,n_patch+1,n_patch+1
        x_atten_all = x_atten[:,0,1:] # bs,n_patch select the attention of only score token
        x_lead = x_lead[:,1:,:] # bs,n_patch+1,d -> bs,n_patch,d
        x_lead,x_pos_masked = self.sortANDgather(x_atten_all,x_lead,x_pos_masked,pure_ways=pure_ways)

        return x_lead,x_pos_masked


class Token_Purification_M(nn.Module):
    def __init__(self, in_dim, pufication_ratio=0.5,  num_head=4):
        '''

        input L+1,D
        out L/r, D
        '''
        super().__init__()
        self.pufication_ratio = pufication_ratio
        # self.fc_expand = nn.Linear(in_dim, out_dim)
        self.embed_dim = in_dim
        self.cross_atten = MultiHeadAttention_atten_only(in_dim, num_heads=num_head)


    def calc_atten_only(self,x):
        self.scale = self.embed_dim ** -0.5
        dots = torch.einsum('bid,bjd->bij', x, x) * self.scale
        attn = dots.softmax(dim=-1)
        return attn

    def sortANDgather(self,atten,input,positin_embedding,pure_ways='atten'):
        def shuffle(t):
            noise = torch.rand(t.shape[0], t.shape[1], device=input.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            t = torch.gather(t, dim=1, index=ids_shuffle)
            return t

        N,L,D = input.shape
        len_keep = int(L*self.pufication_ratio)


        ids_sorted = torch.argsort(atten, dim=1,descending=True)  # large to small # bs,L
        ids_keep_atten = ids_sorted[:, :len_keep]
        ids_keep_NOTatten = ids_sorted[:, len_keep:]

        noise = torch.rand(N, L, device=input.device)  # noise in [0, 1]   [2, 120]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_noise = ids_shuffle[:, :len_keep]

        if pure_ways in ['atten']:
            ids_keep=ids_keep_atten
        elif pure_ways in ['random']:
            ids_keep=ids_keep_noise
        elif pure_ways in ['combine']:
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_atten = shuffle(ids_keep_atten)
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_NOTatten = shuffle(ids_keep_NOTatten)
            ids_keep = torch.cat([ids_keep_atten[:,:len_keep//2],ids_keep_NOTatten[:,len_keep//2:len_keep]],dim=1)
        # print('ids_keep',ids_keep.shape)
        # assert 1>2
        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))

        return x_selected,x_selected_pos

    
    def forward(self, x_lead, x_pos_masked, pure_ways='atten'):
        if pure_ways not in ['atten','random','combine']:
            raise ValueError(f'{pure_ways} not support')
        
        score_emb = x_lead[:,0,:] # bs,n_patch+1,d -> bs,n_patch,d
        score_emb = score_emb.unsqueeze(1)
        # print('score_emb',score_emb.shape)
        x_lead = x_lead[:,1:,:] # bs,n_patch+1,d -> bs,n_patch,d
        score_atten = self.cross_atten(score_emb,key=x_lead) # bs,1,dim  bs,n_head,dim  => bs,n_head,1,n_patch
        score_atten = torch.mean(score_atten,1) #  bs,n_head,1,n_patch ->  bs,1,n_patch
        # print('score_atten',score_atten.shape)
        x_atten_all = score_atten[:,0,:] # bs,n_patch select the attention of only score token

        x_lead,x_pos_masked = self.sortANDgather(x_atten_all,x_lead,x_pos_masked,pure_ways=pure_ways)

        return x_lead,x_pos_masked



class MultiHeadAttention_atten_only(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.content_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, query, key=None):
        h = self.num_heads

        query = self.query_proj(query)
        key = self.content_proj(key)

        q = rearrange(query, 'b n_q (h d) -> b h n_q d', h = h)
        k = rearrange(key, 'b n_k (h d) -> b h n_k d', h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        return attn

class Fast_Purification(nn.Module):
    def __init__(self, in_dim,num_head=8,):
        '''

        input L+1,D
        out L/r, D
        '''
        super().__init__()

        self.embed_dim = in_dim
        self.score_emb = nn.Embedding(1, in_dim).requires_grad_(True)

        self.cross_atten = MultiHeadAttention_atten_only(in_dim, num_heads=num_head)
                    

    def calc_cross_atten_only(self,x,key):
        self.scale = self.embed_dim ** -0.5
        dots = torch.einsum('bid,bjd->bij', x, key) * self.scale
        attn = dots.softmax(dim=-1)
        return attn

    def sortANDgather(self,atten,input,positin_embedding,mask_ratio=0.5,pure_ways='atten'):
        def shuffle(t):
            noise = torch.rand(t.shape[0], t.shape[1], device=input.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            t = torch.gather(t, dim=1, index=ids_shuffle)
            return t

        N,L,D = input.shape
        len_keep = int(L*(1-mask_ratio))


        ids_sorted = torch.argsort(atten, dim=1,descending=True)  # large to small # bs,L
        ids_keep_atten = ids_sorted[:, :len_keep]
        ids_keep_NOTatten = ids_sorted[:, len_keep:]

        noise = torch.rand(N, L, device=input.device)  # noise in [0, 1]   [2, 120]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_noise = ids_shuffle[:, :len_keep]

        if pure_ways in ['atten']:
            ids_keep=ids_keep_atten
        elif pure_ways in ['random']:
            ids_keep=ids_keep_noise
        elif pure_ways in ['combine']:
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_atten = shuffle(ids_keep_atten)
            # print('ids_keep_atten',ids_keep_atten)
            ids_keep_NOTatten = shuffle(ids_keep_NOTatten)
            ids_keep = torch.cat([ids_keep_atten[:,:len_keep//2],ids_keep_NOTatten[:,len_keep//2:len_keep]],dim=1)
        # print('ids_keep',ids_keep.shape)
        # assert 1>2
        x_selected = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_selected_pos = torch.gather(positin_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, positin_embedding.shape[-1]))

        return x_selected,x_selected_pos,ids_keep

    
    def forward(self, x_lead, x_pos_masked,mask_ratio, fast_pure_ways='atten'):
        if fast_pure_ways not in ['atten','random','combine']:
            raise ValueError(f'{fast_pure_ways} not support')

        score_emb = self.score_emb.weight
        score_emb = score_emb.unsqueeze(0).repeat(x_lead.shape[0],1, 1)


        score_atten = self.cross_atten(score_emb,key=x_lead) # bs,1,dim  bs,n_head,dim  => bs,n_head,1,n_patch
        score_atten = torch.mean(score_atten,1) #  bs,n_head,1,n_patch ->  bs,1,n_patch

        x_atten_all = score_atten[:,0,:] # bs,n_patch select the attention of only score token
        # print('x_atten_all',x_atten_all.shape)
        x_lead,x_pos_masked,ids_keep = self.sortANDgather(x_atten_all,x_lead,x_pos_masked,mask_ratio=mask_ratio,pure_ways=fast_pure_ways)
        return x_lead,x_pos_masked,ids_keep


class Model(nn.Module):
    def __init__(self,input_c=1,input_length=5000,patch_size=250,
                leads_input=['I','II_1','II_2','II_3','II_4','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],

                embedding_1d_2d='1d',

                embed_dim = 128,
                local_depth = 9,
                num_head = 8,

                add_pos_before_cross = True,
                cls_cross_depth=3,

                pred_full_II_lead= False,

                normalize_before = False,
                mode_stage = 'self_learning', # self_learning  classfier
                num_classes = 70,

                pufication_module='mlp_purification', # none mlp_purification token_purification
                pufication_ratio=0.5,

                lead_split = True, # True False
                fast_pufication='score_embedding', # random_mask score_embedding
                fast_pufication_head= 4, # 1,4,8
                Token_Purification_M_head = 4,

                pos_emb_style = 'orginal', # orginal, lead_emb, time_emb, lead_time_emb random_lead_emb
                lead_pos_embedding=None,
                lead_embedding_trainable=False,
                time_emb_source = 'orginal', # orginal 12lead_same scartch

                classfierer_type = 'cls_embeeing_seperate',
                ):

        super().__init__()
        self.input_c = input_c
        self.input_len = input_length
        self.patch_size = patch_size
        self.norm_pix_loss = False
        act_layer=nn.GELU
        '''
        default size
        '''
        norm_layer = nn.LayerNorm
        mlp_ratio = 4.


        if embedding_1d_2d == '2d':
            self.linear_proj = PatchEmbed_2D(patch_size=patch_size, embed_dim=embed_dim)
        elif embedding_1d_2d == '1d':
            self.linear_proj = PatchEmbed_1D(input_c,patch_size=patch_size, embed_dim=embed_dim)
        elif embedding_1d_2d == '2d_time_all':
            self.linear_proj = PatchEmbed_2D_time_all(input_c,patch_size=patch_size, embed_dim=embed_dim)
        else:
            assert 1>2,'{} not support'.format(embedding_1d_2d)
        self.embedding_1d_2d = embedding_1d_2d

        self.n_pathch_one_lead = int(input_length // patch_size)

        self.n_pathch4pos = int(5000//patch_size) # postion 4 full lead
        self.leads_input = leads_input
        self.n_pathch_all_lead_full = int(self.n_pathch4pos * 12) #理想情况下 12导联*10s 的patch

        self.add_pos_before_cross = add_pos_before_cross

        '''model stage set'''
        self.mode_stage = mode_stage
        self.pred_full_II_lead = pred_full_II_lead
        if mode_stage == 'self_learning':
            if self.pred_full_II_lead is False:
                n_knowledge_embedding = len(leads_input)
            else:
                n_knowledge_embedding = 12
                self.last_fc_II1 = nn.Linear(embed_dim,int(input_length)) 
                self.last_fc_II2 = nn.Linear(int(input_length),int(input_length*2)) 
                self.last_fc_II3 = nn.Linear(int(input_length*2),int(input_length*4)) 
                self.last_act = act_layer()

            self.last_fc = nn.Linear(embed_dim,input_length) 
        elif mode_stage == 'classfier':
            n_knowledge_embedding = num_classes
            if classfierer_type == 'cls_embeeing_seperate':
                self.last_fc = nn.ModuleList([  
                    nn.Linear(embed_dim, 2)
                    for i in range(num_classes)])
            elif classfierer_type == 'fc':
                self.last_fc = nn.Linear(embed_dim, num_classes)
            else:
                raise ValueError(f'classfierer_type {classfierer_type} not support')
        else:
            raise ValueError(f'mode_stage {mode_stage} not support')
        self.classfierer_type = classfierer_type


        self.pufication_module = pufication_module
        self.fast_pufication = fast_pufication
        self.lead_split = lead_split

        '''purfication module set'''
        if pufication_module == 'mlp_purification':
            self.mlp_purification = MLP_Purification(embed_dim,pufication_ratio=pufication_ratio,act_layer=act_layer)
        elif pufication_module == 'token_purification':
            self.score_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.token_purification = Token_Purification(embed_dim,pufication_ratio=pufication_ratio)
        elif pufication_module == 'token_purification_M':
            self.score_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.token_purification_M = Token_Purification_M(embed_dim,pufication_ratio=pufication_ratio,num_head=Token_Purification_M_head)

        elif pufication_module == 'none':
            pass
        else:
            raise ValueError(f'pufication_module {pufication_module} not support')

        if fast_pufication == 'score_embedding':
            # self.score_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.fast_purification_module = Fast_Purification(embed_dim,num_head=fast_pufication_head)
        elif fast_pufication == 'random_mask':
            pass
        else:
            raise ValueError(f'fast_pufication {fast_pufication} not support')


        self.encoder_layers = nn.ModuleList([
            selfAttention_layer(embed_dim, num_head, mlp_ratio, 
            normalize_before=normalize_before,
            qkv_bias=True,act_layer=act_layer)
            for i in range(local_depth)])
        self.encoder_norm = norm_layer(embed_dim)

        self.knowledge_self_layers = nn.ModuleList([
            selfAttention_layer(embed_dim, num_head, mlp_ratio, 
            normalize_before=normalize_before,
            qkv_bias=True, act_layer=act_layer)
            for i in range(cls_cross_depth)])

        self.knowledge_cross_layers = nn.ModuleList([
            crossAttention_layer(embed_dim, num_head, mlp_ratio, 
                normalize_before=normalize_before,
                qkv_bias=True,act_layer=act_layer)
                for i in range(cls_cross_depth)])

        self.time_emb_source = time_emb_source
        self.knowledge_emb = nn.Embedding(n_knowledge_embedding, embed_dim).requires_grad_(True)
        self.embed_dim = embed_dim

        self.pos_emb_style = pos_emb_style
        # if pos_emb_style in ['lead_emb', 'time_emb', 'lead_time_emb','random_lead_emb']:
        
        if lead_pos_embedding is not None:
            
            lead_embedding = lead_pos_embedding[0]
            self.pos_embed_data = lead_pos_embedding[1]

            n_lead_emb=lead_embedding.size()[0]
            if n_lead_emb != 15:
                raise ValueError(f'n_lead_emb {n_lead_emb} must 15')

            lead_embedding_all_patch = []
            for i in range(lead_embedding.shape[0]): # bs,12,d -> 750,d
                lead_embedding_i = lead_embedding[i].unsqueeze(0).repeat(self.n_pathch_one_lead,1)
                lead_embedding_all_patch.append(lead_embedding_i)
            self.lead_embedding_all_patch = torch.cat(lead_embedding_all_patch,dim=0)
            self.ori_lead_embedding = lead_embedding
            # self.lead_embedding_all_patch = self.expand_lead_emb(lead_embedding)
            self.lead_embedding = nn.Parameter(torch.zeros(1,self.n_pathch_one_lead*15, embed_dim), requires_grad=lead_embedding_trainable)

        else:
            raise ValueError(f'pos_emb_style {pos_emb_style}, lead_embedding is {lead_embedding} ,must have value')


        if self.time_emb_source == 'scartch':
            self.pos_embed = nn.Embedding(self.n_pathch_one_lead*15, embed_dim).requires_grad_(True)
        elif self.time_emb_source in ['orginal','12lead_same']:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_pathch_all_lead_full, embed_dim), requires_grad=False)
        elif self.time_emb_source in ['time_emb']:
            # self.pos_embed = nn.Parameter(torch.zeros(self.n_pathch_one_lead*15, embed_dim), requires_grad=False)
            self.pos_embed = nn.Embedding.from_pretrained(self.pos_embed_data).requires_grad_(False)
        elif self.time_emb_source in ['random']:
            self.pos_embed = nn.Embedding(self.n_pathch_one_lead*15, embed_dim).requires_grad_(False)
        else:
            raise ValueError(f'time_emb_source {time_emb_source} Wrong!')


        self.initialize_weights()
        self.get_leadIdx()

    def get_leadIdx(self):

        limb_leads = ['I','II_1','III','aVR','aVL','aVF']
        chest_leads = ['V1','V2','V3','V4','V5','V6']
        full_II_leads = ['II_1','II_2','II_3','II_4']


        
        self.limb_leads_idx = []
        self.chest_leads_idx = []
        self.full_II_leads_idx = []

        self.gather_limb_leads_idx = []
        self.gather_chest_leads_idx = []
        self.gather_full_II_leads_idx = []

        self.leads_default_order = {
            'I': 0,'I I': 1,'III': 2,'aVR': 3,'aVL': 4,'aVF': 5,
            'V1': 6,'V2': 7,'V3': 8,'V4': 9,'V5': 10,'V6': 11,
            'II_1': 1,'II_2': 1,'II_3': 1,'II_4': 1,
        }

        if self.time_emb_source in ['orginal','scartch','time_emb','random']:
            self.leads_info = {
                'I': [0,int(self.n_pathch4pos*0.25)],
                'I I': [0,int(self.n_pathch4pos*0.25)],
                'III': [0,int(self.n_pathch4pos*0.25)],
                
                'aVR': [int(self.n_pathch4pos*0.25),int(self.n_pathch4pos*0.5)],
                'aVL': [int(self.n_pathch4pos*0.25),int(self.n_pathch4pos*0.5)],
                'aVF': [int(self.n_pathch4pos*0.25),int(self.n_pathch4pos*0.5)],

                'V1': [int(self.n_pathch4pos*0.5),int(self.n_pathch4pos*0.75)],
                'V2': [int(self.n_pathch4pos*0.5),int(self.n_pathch4pos*0.75)],
                'V3': [int(self.n_pathch4pos*0.5),int(self.n_pathch4pos*0.75)],

                'V4': [int(self.n_pathch4pos*0.75),self.n_pathch4pos*1],
                'V5': [int(self.n_pathch4pos*0.75),self.n_pathch4pos*1],
                'V6': [int(self.n_pathch4pos*0.75),self.n_pathch4pos*1],

                'II_1': [0,int(self.n_pathch4pos*0.25)],
                'II_2': [int(self.n_pathch4pos*0.25),int(self.n_pathch4pos*0.5)],
                'II_3': [int(self.n_pathch4pos*0.5),int(self.n_pathch4pos*0.75)],
                'II_4': [int(self.n_pathch4pos*0.75),self.n_pathch4pos*1],
            }
        elif self.time_emb_source == '12lead_same':
            self.leads_info = {
                'I': [0,int(self.n_pathch4pos*0.25)],
                'I I': [0,int(self.n_pathch4pos*0.25)],
                'III': [0,int(self.n_pathch4pos*0.25)],
                
                'aVR': [0,int(self.n_pathch4pos*0.25)],
                'aVL': [0,int(self.n_pathch4pos*0.25)],
                'aVF': [0,int(self.n_pathch4pos*0.25)],

                'V1': [0,int(self.n_pathch4pos*0.25)],
                'V2': [0,int(self.n_pathch4pos*0.25)],
                'V3': [0,int(self.n_pathch4pos*0.25)],

                'V4': [0,int(self.n_pathch4pos*0.25)],
                'V5': [0,int(self.n_pathch4pos*0.25)],
                'V6': [0,int(self.n_pathch4pos*0.25)],

                'II_1': [0,int(self.n_pathch4pos*0.25)],
                'II_2': [int(self.n_pathch4pos*0.25),int(self.n_pathch4pos*0.5)],
                'II_3': [int(self.n_pathch4pos*0.5),int(self.n_pathch4pos*0.75)],
                'II_4': [int(self.n_pathch4pos*0.75),self.n_pathch4pos*1],
            }
        else:
            raise ValueError(f'time_emb_source {self.time_emb_source} not support')


        for idx,leads_i in enumerate(self.leads_input):
            lead_idx = self.leads_default_order[leads_i]
            start_local,end_local = self.leads_info[leads_i]
            start_global = start_local+lead_idx*self.n_pathch4pos
            end_global = end_local+lead_idx*self.n_pathch4pos
            # print('lead_i {}, start_global {}, end_global {}'.format(leads_i,start_global,end_global))
            for gather_idx_i in range(start_global,end_global):
                # print('gather_idx_i',gather_idx_i)
                if leads_i in limb_leads: 
                    self.gather_limb_leads_idx.append(gather_idx_i)
                elif leads_i in chest_leads: 
                    self.gather_chest_leads_idx.append(gather_idx_i)
                elif leads_i in full_II_leads: 
                    self.gather_full_II_leads_idx.append(gather_idx_i)

            if leads_i in limb_leads: 
                self.limb_leads_idx.append(idx)
            elif leads_i in chest_leads: 
                self.chest_leads_idx.append(idx)
            elif leads_i in full_II_leads: 
                self.full_II_leads_idx.append(idx)
            else:
                raise ValueError(f'{leads_i} not support')

        if self.lead_split == True:
            self.leads_idx = [self.limb_leads_idx,self.chest_leads_idx,self.full_II_leads_idx]
            if self.time_emb_source in ['scartch','time_emb','random']:
                self.leads_pos_gather_idx = [[i for i in range(0,self.n_pathch_one_lead*6)],
                                            [i for i in range(self.n_pathch_one_lead*6,self.n_pathch_one_lead*12)],
                                            [i for i in range(self.n_pathch_one_lead*12,self.n_pathch_one_lead*15)],]
            else:
                self.leads_pos_gather_idx = [self.gather_limb_leads_idx,self.gather_chest_leads_idx,self.gather_full_II_leads_idx]

        else:
            # all input 全输入
            self.leads_idx = [0,]
            self.leads_pos_gather_idx = [self.__get_default_pos_loc()]
            # if self.time_emb_source == 'scartch':
            #     self.leads_pos_gather_idx = [[i for i in range(0,self.n_pathch_one_lead*15)+1]]


    def initialize_weights(self):
        # initialization
        if self.pos_emb_style in ['orginal']:
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size_w=self.n_pathch4pos, grid_size_h=12, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))            
        else:
            #only consider time series
            if self.time_emb_source == 'orginal':
                pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size_w=self.n_pathch4pos, grid_size_h=1, cls_token=False)
                pos_embeds = [pos_embed for i in range(12)]
                pos_embeds = np.concatenate(pos_embeds)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embeds).float().unsqueeze(0))            
            
                # if self.mode_stage == 'classfier':
            if self.pos_emb_style in ['random_lead_emb']:
                self.lead_embedding.data.copy_(torch.rand(1,self.n_pathch_one_lead*15, self.embed_dim))
            else:
                self.lead_embedding.data.copy_(self.lead_embedding_all_patch)

            # if not torch.equal(self.lead_embedding,self.lead_embedding_all_patch):
            #     raise ValueError(f'lead_embedding {self.lead_embedding} != lead_embedding_all_patch {self.lead_embedding_all_patch.shape}')

        if self.embedding_1d_2d in ['2d','2d_time_all']:
            w = self.linear_proj.projs.weight.data
            # w = self.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def __get_default_pos_loc(self):
        '''
        获取对应的 pos idx，注意不是pos
        row 为 lead
        column 为时间轴
        '''
        gather_idx = []
        for i in range(len(self.leads_input)):
            lead_i = self.leads_input[i]

            lead_idx = self.leads_default_order[lead_i]

            # 如果只输入ii导联
            if len(self.leads_input) == 4 and lead_i in ['II_1','II_2','II_3','II_4']:
                lead_idx = 0
            # if lead_i in ['II_1','II_2','II_3','II_4']: #最后一个长ii
            #     lead_idx = 1
            # else:
            #     lead_idx = leads_default_order.index(lead_i)
            
            start_local,end_local = self.leads_info[lead_i]
            start_global = start_local+lead_idx*self.n_pathch4pos
            end_global = end_local+lead_idx*self.n_pathch4pos
            # print('lead_i {}, start_global {}, end_global {}'.format(lead_i,start_global,end_global))
            for gather_idx_i in range(start_global,end_global):
                # print('gather_idx',gather_idx)
                gather_idx.append(gather_idx_i)
        return gather_idx


    def patchify(self, input_x):
        """
        input_x: (N, n_c, W)
        x: (N, L*n_c, patch_size)
        """
        p = self.patch_size
        w = self.n_pathch_one_lead #n_patch
        h = self.input_c
        assert input_x.shape[2] % p == 0
        x = input_x.reshape(shape=(input_x.shape[0], h * w, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L*n_c, patch_size)
        input_x: (N, n_c, W)
        """
        p=self.patch_size
        w = self.n_pathch_one_lead #n_patch
        h = self.input_c
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, 1, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        input_x = x.reshape(shape=(x.shape[0], self.input_c, w * p))
        return input_x


    def random_masking(self, x, pos_embeding_loc, mask_ratio,n_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if n_mask is None:
            len_keep = int(L * (1 - mask_ratio))
        else:
            len_keep = L-n_mask
            assert len_keep > 0,'L {} - n_mask {} = len_keep {} <=0'.format(L,n_mask,len_keep)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]   [2, 120]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_pos_masked = torch.gather(pos_embeding_loc, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked,x_pos_masked, mask, ids_restore,ids_shuffle


    def expand_lead_emb(self,lead_embedding):
        '''lead,d -> lead*n_patch,d '''
        lead_embedding_all_patch = []
        for i in range(lead_embedding.shape[0]): # bs,12,d -> 750,d
            lead_embedding_i = lead_embedding[i].unsqueeze(0).repeat(1,self.n_pathch_one_lead,1)
            lead_embedding_all_patch.append(lead_embedding_i)
        return torch.cat(lead_embedding_all_patch,dim=1)



    def forward(self, input,mask_ratio=0.,
                pure_ways='atten',
                fast_pure_ways='atten',
                register_hook=False):

        bs,n_leads,n_length = input.shape

        # print('input',input.shape)
        x_leads_all = []
        x_pos_mask_all = []
        for leads_idx,pos_idx in zip(self.leads_idx,self.leads_pos_gather_idx):
            if self.lead_split is True:
                input_lead = input[:,leads_idx,:]
            else:
                input_lead = input

            x_lead = self.linear_proj(input_lead) 

            '''可训练的time_emb'''
            if self.time_emb_source in ['scartch','time_emb','random']:
                pos_embed = self.pos_embed.weight
                pos_embeding_loc = pos_embed.unsqueeze(0).repeat(bs,1,1)
                # print('pos_embeding_loc before',pos_embeding_loc.shape)
                # print('pos_idx',pos_idx,len(pos_idx))
                if self.lead_split is True: pos_embeding_loc = pos_embeding_loc[:,pos_idx,:]
                # print('pos_embeding_loc after',pos_embeding_loc.shape)
            else:
                pos_embed_i = self.pos_embed[0][pos_idx]
                pos_embeding_loc = pos_embed_i.repeat(bs, 1, 1)
            
            # x_lead+=pos_embeding_loc

            if self.pos_emb_style in ['orginal']:                    
                x_lead+=pos_embeding_loc

            if self.pos_emb_style in ['lead_time_emb','time_emb','random_lead_emb']:
                # if self.time_emb_source == 'scartch':
                if self.time_emb_source in ['scartch','time_emb','random']:
                    x_lead+=pos_embeding_loc
                else:
                    x_lead+=pos_embeding_loc*0.3


            if self.pos_emb_style in ['lead_emb','random_lead_emb','lead_time_emb']:
                if self.mode_stage == 'self_learning':
                    knowledge_emb = self.knowledge_emb.weight
                    
                    if knowledge_emb.shape[0] == 12:
                        knowledge_emb = torch.cat([knowledge_emb[0].unsqueeze(0),
                            knowledge_emb[1].unsqueeze(0),
                            knowledge_emb[1].unsqueeze(0),
                            knowledge_emb[1].unsqueeze(0),
                            knowledge_emb[1].unsqueeze(0),
                            knowledge_emb[2:]],dim=0)
                        # print('knowledge_emb',len(self.leads_idx), len(self.leads_pos_gather_idx), knowledge_emb.shape)

                    if self.lead_split is True:
                        lead_embedding_i = knowledge_emb[leads_idx] # 12,d -> lead,d
                        pos_lead_emb = self.expand_lead_emb(lead_embedding_i)
                        pos_lead_emb = pos_lead_emb.repeat(bs,1,1)
                    else:
                        # pos_lead_emb = self.lead_embedding.repeat(bs, 1, 1)
                        lead_embedding_all_patch = []
                        for i in range(input.shape[1]): # bs,12,d -> 750,d
                            lead_embedding_i = knowledge_emb[i].unsqueeze(0).repeat(self.n_pathch_one_lead,1)
                            lead_embedding_all_patch.append(lead_embedding_i)
                        knowledge_emb = torch.cat(lead_embedding_all_patch,dim=0)
                        pos_lead_emb = knowledge_emb.unsqueeze(0).repeat(bs,1, 1)
                else:
                    if self.lead_split is True:
                        # print('self.lead_embedding',self.lead_embedding)
                        # assert 1>2

                        # print('self.lead_embedding',self.lead_embedding,self.lead_embedding.shape)
                        # index_tmp = [0,1,5,6,7,8]

                        # print('self.lead_embedding[0][index_tmp]',self.lead_embedding[0][index_tmp],self.lead_embedding[0][index_tmp].shape)

                        # print('self.lead_embedding',self.lead_embedding.shape)
                        # print('self.lead_embedding[0]',self.lead_embedding[0].shape)
                        # lead_embedding_i = self.lead_embedding[0][1]
                        # leads_idx = [1,2,3]
                        # print('leads_idx',leads_idx)
                        # print('lead_embedding_i',lead_embedding_i.shape)
                        
                        lead_embedding_i = self.lead_embedding[0][leads_idx] # 12,d -> lead,d
                        # print('lead_embedding_i',lead_embedding_i.shape)

                        pos_lead_emb = self.expand_lead_emb(lead_embedding_i)
                        pos_lead_emb = pos_lead_emb.repeat(bs,1,1)
                    else:
                        pos_lead_emb = self.lead_embedding.repeat(bs, 1, 1)
                    
                    # print('self.lead_embedding',self.lead_embedding)
                # print('x_lead',x_lead.shape)
                # print('pos_lead_emb',pos_lead_emb.shape)
                x_lead+=pos_lead_emb

            if self.fast_pufication == 'random_mask':
                x_lead,x_pos_masked, mask, ids_restore,ids_shuffle = self.random_masking(x_lead, pos_embeding_loc,mask_ratio)
            elif self.fast_pufication == 'score_embedding':
                x_lead,x_pos_masked,idx_keep = self.fast_purification_module(x_lead, pos_embeding_loc,mask_ratio,fast_pure_ways=fast_pure_ways)
            else:
                raise ValueError(f'fast_pufication {self.fast_pufication} not support')

            if self.pufication_module == 'token_purification':
                cls_tokens = repeat(self.score_token, '() n d -> b n d', b = bs)
                x_lead = torch.cat((cls_tokens, x_lead), dim=1)

            for encoder_layer in self.encoder_layers:
                x_lead = encoder_layer(x_lead, register_hook=register_hook)
            
            # print('x_lead',x_lead.shape)
            if self.pufication_module == 'mlp_purification':
                x_lead,x_pos_masked = self.mlp_purification(x_lead,x_pos_masked,pure_ways=pure_ways)
            elif self.pufication_module == 'token_purification':
                x_lead,x_pos_masked = self.token_purification(x_lead,x_pos_masked,pure_ways=pure_ways)
            elif self.pufication_module == 'token_purification_M':
                x_lead,x_pos_masked = self.token_purification_M(x_lead,x_pos_masked,pure_ways=pure_ways)

            # print('pufication_module',self.pufication_module)
            # print(f'x_lead after purification {self.pufication_module}',x_lead.shape)
            # print('x_pos_masked',x_pos_masked.shape)
            # print('-'*10)
            # assert 1>2

            x_leads_all.append(x_lead)
            x_pos_mask_all.append(x_pos_masked)

        x_leads_all = torch.cat(x_leads_all,dim=1)
        x_pos_mask_all = torch.cat(x_pos_mask_all,dim=1)
        # x_leads_all = self.encoder_norm(x_leads_all)

        if self.add_pos_before_cross is True:
            x_leads_all+=x_pos_mask_all

        # assert 1>2
        if self.classfierer_type != 'fc':
            knowledge_emb = self.knowledge_emb.weight
            knowledge_emb = knowledge_emb.unsqueeze(0).repeat(bs,1, 1)
            # knowledge_emb = knowledge_emb.repeat(bs,1, 1)
            for self_atten_layer,cross_atten_layer in zip(self.knowledge_self_layers, self.knowledge_cross_layers):
                knowledge_emb = self_atten_layer(knowledge_emb, register_hook=register_hook) # 15,64
                # print('knowledge_emb self',knowledge_emb.shape)
                knowledge_emb = cross_atten_layer(knowledge_emb, key=x_leads_all, value=x_leads_all, register_hook=register_hook) ## 15,64
                # print('knowledge_emb cross',knowledge_emb.shape)

        if self.mode_stage == 'self_learning':

            if self.pred_full_II_lead is False:
                pred = self.last_fc(knowledge_emb) # [bs, 15, 64] -> [bs, 15, 1250]
            else:

                knowledge_emb_II = knowledge_emb[:,0,:]
                knowledge_emb_II = self.last_fc_II1(knowledge_emb_II)
                knowledge_emb_II = self.last_act(knowledge_emb_II)
                knowledge_emb_II = self.last_fc_II2(knowledge_emb_II)
                knowledge_emb_II = self.last_act(knowledge_emb_II)
                pred_II = self.last_fc_II3(knowledge_emb_II)
                pred_II = pred_II.unsqueeze(1)
                pred_II = pred_II.reshape(bs,4,pred_II.shape[-1]//4)

                knowledge_emb_rest = knowledge_emb[:,1:,:]
                pred_rest = self.last_fc(knowledge_emb_rest) # [bs, 15, 64] -> [bs, 15, 1250]

                pred = torch.cat(
                    [
                        pred_rest[:,0,:].unsqueeze(1),
                        pred_II,
                        pred_rest[:,1:,:],
                    ],dim=1
                )

        elif self.mode_stage == 'classfier':
            if self.classfierer_type == 'cls_embeeing_seperate':
                x_concats = []
                for cls_idx,class_head in enumerate(self.last_fc):
                    x_i = class_head(knowledge_emb[:,cls_idx,:]) #  [bs, 512] -> [bs, 2]
                    x_i = x_i.unsqueeze(1) #  [bs, 2] -> [bs, 1, 2]
                    x_concats.append(x_i)
                pred = torch.cat(x_concats, 1)  #  [bs, 60, 2]
            else:
                x_features = x_leads_all.mean(dim=1)  # global pool without cls token 
                pred = self.last_fc(x_features) # # [bs, n_dim] -> [bs,n_cls]
        return pred




