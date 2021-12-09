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




## References

[^1]: http://einops.rocks/pytorch-examples.html
[^2]: https://github.com/arogozhnikov/einops
