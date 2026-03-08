---
title: "Einops - Part 2: Deep Learning Architectures"
tags:
  - deep-learning
  - einops
  - pytorch
summary: "Building ConvNets and Multi-Head Attention with einops  -  comparing traditional PyTorch vs einops implementations."
---

# Deep Learning Architectures using EINOPS

In this section we will be rewriting the building blocks of deep learning
in both the traditional `PyTorch` way as well as using `einops` library.

## Imports

```python
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
```

## Simple ConvNet

### Using only PyTorch

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)                          # <-- RESHAPE
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

### Using einops

```python
class ConvNetEinops(nn.Module):
    def __init__(self):
        super(ConvNetEinops, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = rearrange(x, "b c h w -> b (c h w)")     # <-- einops RESHAPE
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

## Multi-Head Attention

### Using only PyTorch

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

    def forward(self, q, k, v, mask=None):
        B, Lq, Lk, Lv = q.size(0), q.size(1), k.size(1), v.size(1)
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        qs = self.w_qs(q).view(B, Lq, n_head, d_k)   # <-- RESHAPE
        ks = self.w_ks(k).view(B, Lk, n_head, d_k)
        vs = self.w_vs(v).view(B, Lv, n_head, d_v)

        qs = qs.permute(0, 2, 1, 3)                   # <-- TRANSPOSE
        ks = ks.permute(0, 2, 1, 3)
        vs = vs.permute(0, 2, 1, 3)
        # ... rest of attention
```

### Using einops

```python
class MultiHeadAttentionEinops(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

    def forward(self, q, k, v, mask=None):
        qs = rearrange(self.w_qs(q), "b l (head k) -> b head l k", head=self.n_head)
        ks = rearrange(self.w_ks(k), "b t (head k) -> b head t k", head=self.n_head)
        vs = rearrange(self.w_vs(v), "b t (head v) -> b head t v", head=self.n_head)
        # ... rest of attention
```