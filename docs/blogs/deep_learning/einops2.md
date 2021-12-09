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

???+ done "ConvNet using EINOPS + PyTorch"
    Implementing the same above ConvNet using **`einops`** & `PyTorch`
    
    ```python
    
    ```


## References

[^1]: http://einops.rocks/pytorch-examples.html
[^2]: https://github.com/arogozhnikov/einops
