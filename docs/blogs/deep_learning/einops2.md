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
    ```python
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
            x = self.relu(self.conv4(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x
    ```

## References

[^1]: http://einops.rocks/pytorch-examples.html
[^2]: https://github.com/arogozhnikov/einops
