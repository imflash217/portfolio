# Popular Deep Learning Architectures using EINOPS

In this section we will be rewriting the building blocks of deep learning
in both the traditional `PyTorch` way as well as using `einops` library.

## Imports

Firstly, we will import the necessary libraries to be used.

```python hl_lines="10-11"
## importing necessary libraries

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
```

## Simple ConvNet

???+ danger "Using only PyTorch"
    Here is an implementation of a simple **ConvNet** using only **`PyTorch`**
    without `einops`.

    ```python hl_lines="29"
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(20, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)
            
            x = self.conv2(x)
            x = self.conv2_drop(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = x.view(-1, 320)
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    ## Instantiating the ConvNet class
       
    conv_net_old = ConvNet()
    ```

???+ done "Using EINOPS + PyTorch"
    Implementing the same above ConvNet using **`einops`** & `PyTorch`
    
    ```python hl_lines="9"
    conv_net_new = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Dropout2d(),
        Rearrange("b c h w -> b (c h w)"),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(50, 10),
        nn.LogSoftmax(dim=1)
    )
    ```

???+ quote "Why prefer EINOPS implementation?"
    Following are the reasons to prefer the new implementation:

    - [x] In the original code, if the input is changed and the **`batch_size`** 
            is divisible by 16 (which usually is), we will get something senseless after reshaping.
        - [ ] :rotating_light: The new code using **`einops`** explicitly raise ERROR in the above scenario. Hence better!!
    - [x] We won't forget to use the flag **`self.training`** with the new implementation.
    - [x] Code is straightforward to read and analyze.
    - [x] **`nn.Sequential`** makes printing/saving/passing trivial. 
            And there is no need in your code to **load** the model (which also has lots of benefits).
    - [x] Don't need **`logsoftmax`**? Now, you can use **`conv_net_new[-1]`**. 
            Another reason to prefer **`nn.Sequential`**
    - [x] ... And we culd also add **inplace `ReLU`**


## Super-resolution

???+ danger "Only PyTorch"
    ```python hl_lines="9-10 16"
    class SuperResolutionNetOLD(nn.Module):
        def __init__(self, upscale_factor):
            super(SuperResolutionNetOLD, self).__init__()
            
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x
    ```

???+ done "Using EINOPS"
    ```python hl_lines="10"
    def SuperResolutionNetNEW(upscale_factor):
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1),
            Rearrange("b (h2 w2) h w -> b (h h2) (w w2)", h2=upscale_factor, w2=upscale_factor)
        )
    ```

???+ quote "Improvements over the old implementation"
    - [x] No need in special instruction **`pixel_shuffle`** (& the result is transferrable b/w the frameworks)
    - [x] Output does not contain a fake axis (& we could do the same for the input)
    - [x] inplace **`ReLU`** used now. For high resolution images this becomes critical
            and saves a lot of memory.
    - [x] and all the benefits of **`nn.Sequential`**

## Gram Matrix / Style Transfer

Restyling Graam Matrix for style transfer.

???+ danger "Original Code using ONLY PyTorch"
    The original code is already very good. First line shows what kind of input is expected.
    
    ```python
    def gram_matrix_old(y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    ```

???+ done "Using EINSUM"
    ```python hl_lines="3"
    def gram_matrix_new(y):
        b, c, h, w = y.shape
        return torch.einsum("bchw, bdhw -> bcd", [y, y]) / (h * w)
    ```

???+ quote "Improvements"
    **`einsum`** operations should be read like:

    - [x] For each batch & each pair of channels we sum over **`h`** and **`w`**.
    - [x] The normalization is also changed, because that's how **Gram Matrix** is defined.
            Else we should call it **Normalized Gram Matrix** or alike.


## Recurrent Models (RNNs)

???+ danger "ONLY PyTorch"
    ```python hl_lines="14-15"
    class RNNModelOLD(nn.Module):
        """Container module with an ENCODER, a RECURRENT module & a DECODER module"""
        def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(RNNModelOLD, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        
        def forward(self, input, hidden):
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    ```


???+ done "Using EINOPS"
    ```python hl_lines="11 14-15"
    def RNNModelNEW(nn.Module):
        """Container module with an ENCODER, RNN & a DECODER modules."""
        def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(RNNModelNEW, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        
        def forward(self, input, hidden):
            t, b = input.shape[:2]
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = rearrange(self.drop(output), "t b nhid -> (t b) nhid")
            decoded = rearrange(self.decoder(output), "(t b) token -> t b token", t=t, b=b)
            return decoded, hidden
    ```



## References

[^1]: http://einops.rocks/pytorch-examples.html
[^2]: https://github.com/arogozhnikov/einops
