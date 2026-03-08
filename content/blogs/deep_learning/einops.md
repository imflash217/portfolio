---
title: "Einops - Part 1"
tags:
  - deep-learning
  - einops
  - tensor
summary: "Introduction to einops  -  readable tensor operations with rearrange, reduce, and repeat replacing cryptic transposes."
---

## Part-1

### Welcome to einops

1. We don't write
    ```python
    y = x.transpose(0,2,3,1)
    ```
2. We write comprehensible code
    ```python
    y = einops.rearrange(x, "b c h w -> b h w c")
    ```
3. `einops` supports widely used tensor packages viz.
    `numpy`, `pytorch`, `tensorflow`, `chainer`, `gluon`
    and **extends** them.

### What's in this tutorial?

1. **Fundamentals**: reordering, composition, and decomposition of tensors.
2. **Operations**: `rearrange`, `reduce`, `repeat`
3. How much can you do with a **single** operation?

### Preparations
```python
import numpy
from utils import display_np_arrays_as_images
display_np_arrays_as_images()
```

### Load a batch of images
```python
## there are 6 images of shape 96x96
## with 3 color channels packed as tensors
images = np.load("./resources/test_images.npy", allow_pickle=False)
print(images.shape)  ## (6, 96, 96, 3)
```

### Composition of axes
```python
from einops import rearrange

## composing the first two axes together
## i.e. 6 images of shape (96, 96, 3) are rearranged
## into a single image of shape (6*96, 96, 3)
rearrange(images, "b h w c -> (b h) w c").shape  ## (576, 96, 3)
```

### Decomposition of axes
```python
## decomposition is the reverse of composition
rearrange(images, "b h w c -> b h w c").shape  ## (6, 96, 96, 3)
```

### Order of axes matters
```python
## moving color channel to the 2nd position
rearrange(images, "b h w c -> b c h w").shape  ## (6, 3, 96, 96)
```