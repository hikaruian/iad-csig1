"""INP-Former building blocks (faithful to luow23/INP-Former)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PrototypeAttention(nn.Module):
    """Cross-attention: image tokens query intrinsic normal prototypes."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        b, n, c = x.shape
        proto_n = prototype_token.shape[1]
        q = self.q(x).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(b, proto_n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AggregationAttention(nn.Module):
    """INP extractor: prototype queries attend over image tokens."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        b, t, c = x.shape
        _, n, _ = y.shape
        q = self.q(x).reshape(b, t, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, t, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AggregationBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregationAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PrototypeBlock(nn.Module):
    """INP-guided decoder block. Attention has no residual — reconstruction is forced through INPs."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrototypeAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, prototype, return_attention=False):
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
