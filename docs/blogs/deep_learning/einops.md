<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

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
images = np.load("./resources/tes_images.npy", allow_pickle=False)

print(images.shape, images.dtype)   ## (6, 96, 96, 3), float64
```

```python
## display the 1st image (whole 4d tensor can't be rendered)
images[0]
```
<figure markdown class="card">
![...](../../../assets/blogs/deep_learning/einops/images_0.png)
</figure>

```python
images[1]
```
<figure markdown class="card">
![...](../../../assets/blogs/deep_learning/einops/images_1.png)
</figure>

**We will use three opeartions:** `rearrange`, `reduce`, `repeat`
```python
from einops import rearrange, reduce, repeat
```

### Meet "rearrange"

???+ done "rearrange"
    As its name suggests; it rearranges elements. Below, we swap `height` and `width`.

    In other words, below we **transpose** first two axes/dimensions.
    ```python
    rearrange(images[0], "h w c -> w h c")
    ```
    <figure markdown class=card>
        ![images_2](../../../assets/blogs/deep_learning/einops/images_2.png)
    </figure>

### Composition of axes

Transposition is very common and useful; but let's move to other 
operations provided by `einops`

???+ done "composition using `rearrange()` : height"
    `einops` allows seamlessly composing `batch` and `height` to a `new height` dimension.

    Below we just rendered all images in the 4D tensor by collapsing it to a 3D tensor.
    ```python
    rearrange(images, "b h w c -> (b h) w c")
    ```
    <figure markdown class="card">
        ![images_3](../../../assets/blogs/deep_learning/einops/images_3.png)
    </figure>


???+ danger "composition using `rearrange()`: width"
    `einops` allows seamlessly composing `batch` and `width` to a `new width` dimension.

    Below we just rendered all images in the 4D tensor by collapsing it to a 3D tensor.
    ```python
    rearrange(images, "b h w c -> h (b w) c")
    ```
    <figure markdown class="card">
        ![images_4](../../../assets/blogs/deep_learning/einops/images_4.png)
    </figure>

Resulting dimensions are computed very simply. 
**Length of any newly computed axes/dimension is a product of its components**
```python
## [6, 96, 96, 3] -> [96, (6*96), 3]
a = rearrange(images, "b h w c -> h (b w) c")
a.shape
```
```
(96, 576, 3)
```

We can compose more than 2 axes/dimensions.
Let's **flatten** the whole 4D array into a 1D array.
The resulting 1D array contains as many elements as the original 4D array.

```python
## [6, 96, 96, 3] -> [(6*96*96*3)]
a = rearrange(images, "b h w c -> (b h w c)")
a.shape
```
```
(165888, )
```

### Decomposition of axes

**Decomposition** is the inverse process of composition. 

**It represents an existing axis as a combination of new axes**.

Several decompositions are possible. Some examples are shown below:

???+ danger "Combining _composition_ and _decomposition_"
    Combining composition & decomposition
    ```python
    ## here b1=2, decomposes b=6 into "b1=2" and "b2=3"
    ## keeping b = b1*b2
    a = rearrange(images, "(b1 b2) w h c -> b1 b2 w h c", b1=2)
    a.shape     ## (2, 3, 96, 96, 3)
    ```
    ```
    (2, 3, 96, 96, 3)
    ```

???+ done "An example"
    Combining composition & decomposition
    ```python
    ## here b1=2, decomposes b=6 into "b1=2" and "b2=3"
    ## keeping b = b1*b2
    a = rearrange(images, "(b1 b2) w h c -> (b1 h) (b2 w) c", b1=2)

    a.shape     ## (2*96, 3*96, 3)
    a
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_5.png)
    </figure>

???+ danger "Another combination"
    Combining composition & decomposition
    ```python
    ## here b1=2, decomposes b=6 into "b1=2" and "b2=3"
    ## keeping b = b1*b2
    a = rearrange(images, "(b1 b2) w h c -> (b2 h) (b1 w) c", b1=2)

    a.shape     ## (3*96, 2*96, 3)
    a
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_6.png)
    </figure>

???+ done "Another example: `width_to_height`"
    Move part of the `width` dimension to `height`

    We should call this `width_to_height` as the image `width` shrunk by 2 and `height` incresed by 2.

    **But all pixels are same!!!**

    ```python
    a = rearrange(images, "b h (w w2) c -> (h w2) (b w) c", w2=2)

    a.shape     ## (96*2, 6*48, 3)
    a
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_7.png)
    </figure>

???+ done "Another example: `heigh_to_width`"
    Move part of the `height` dimension to `width`

    We should call this `height_to_width` as the image `height` shrunk by 2 and `width` incresed by 2.

    **But all pixels are same!!!**

    ```python
    a = rearrange(images, "b (h h2) w c -> (b h) (w h2) c", h2=2)

    a.shape     ## (6*48, 96*2, 3)
    ```

### Order of axes matter

The order of axes in composition and decomposition is of prime importance. 
It affects the way data is being transposed. Below examples show the impacts.

???+ danger "An example"
    ```python
    a = rearrange(images, "b h w c -> h (b w) c")       ## notice the ordering of (b w)
    a.shape                                             ## (96, 6*96, 3)
    a
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_8.png)
    </figure>

    v/s

    ```python
    b = rearrange(images, "b h w c -> h (w b) c")       ## notice the ordeing of (w b)
    b.shape                                             ## (96, 96*6, 3)
    b
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_9.png)
    </figure>

    **Though the shapes of both `a` and `b` are same but the ordering of pixels are different.**

    **`RULE`**: The rule of importance is just as for digits. 
    The **leftmost** digit is **most significant**.
    Neighboring number differ in _rightmost_ axis.

    ------------------------------------------------------
    
    What will happen if `b1` and `b2` are _reordered_ before composing to `width` 
    (as shown in examples below):
    ```python
    rearrange(images, "(b1 b2) h w c -> h (b1 b2 w) c", b1=2)     ## produces "einops"
    rearrange(images, "(b1 b2) h w c -> h (b2 b1 w) c", b1=2)     ## prodices "eoipns"
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_10.png)
    </figure>

### Meet "reduce"

In `einops` we don't need to guess what happened (like below)
```python
x.mean(-1)
```
Because we write clearly what happened (as shown below)
```python
import einops.reduce

reduce(x, "b h w c -> b h w", "mean")
```
If an axis was not present in the output definition --you guessed it -- it was **reduced**

???+ done "Average over batch"
    Average over batch
    ```python
    u = reduce(images, "b h w c -> h w c", "mean")      ## reduce using "mean" across the "batch" axis
    u.shape                                             ## (96, 96, 3)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_11.png)
    </figure>

    The above code is similar to the standard code (without `einops`) as shown below
    ```python
    u = images.mean(axis=0)     ## find mean across the "batch" dimension 
    u.shape                     ## (96, 96, 3)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_12.png)
    </figure>

    But, the code with `einops` is much more readable and states the operations clearly.

???+ danger "Reducing over multiple axes"
    Example of reducing over several dimensions.
    
    Besides `"mean"`, there are also `"min"`, `"max"`, `"sum"`, `"prod"`
    ```python
    u = reduce(images, "b h w c -> h w", "min")     ## redce across "batch" & "channel" axes
    u.shape                                         ## (96, 96)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_13.png)
    </figure>

### Mean-pooling

???+ done "Mean pooling with 2x2 kernel"
    Image is split into 2x2 patch and each path is avergaed
    ```python
    u = reduce(images, "b (h h2) (w w2) c -> h (b w) c", "mean", h2=2, w2=2)
    u.shape         ## (48, 6*48, 3)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_14.png)
    </figure>

### Max-pooling

???+ danger "max-pooling with 2x2 kernel"
    Image is split into 2x2 patch and each patch is max-pooled
    ```python
    u = reduce(images, "b (h h2) (w w2) c -> h (b w) c", "max", h2=2, w2=2)
    u.shape         ## (49, 6*48, 3)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_15.png)
    </figure>

???+ danger "yet another example"
    ```python
    u = reshape(images, "(b1 b2) h w c -> (b2 h) (b1 w)", "mean", b1=2)
    u.shape         ## (3*96, 2*96)
    u
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_16.png)
    </figure>

### Stack & Concatenate

```python
## rearrange can also take care of lists of arrays with the same shapes

x = list(images)

## Case-0: We can use the "list-axis" as 1st axis ("b") and rest of the axes stays as usual
x0 = rearrange(x, "b h w c -> b h w c")
x0.shape                                    ## (6, 96, 96, 3)
```

```python
##----------------------------------------------------------------------------##
## case-1: But the new axis can appear in any place
x1 = rearrange(x, "b h w c -> h w c b")
x1.shape                                    ## (96, 96, 3, 6)

## This is equivalent to using `numpy.stack`
x11 = numpy.stack(x, axis=3)
x11.shape                                   ## (96, 96, 3, 6)
```

```python
##----------------------------------------------------------------------------##
## Case-2: ....Or we can also concatenate along axes
x2 = rearrange(x, "b h w c -> h (b w) c")
x2.shape                                    ## (96, 6*96, 3)

## This is equivalent to using `numpy.concatenate`
x22 = numpy.concatenate(x, axis=1)
x22.shape                                   ## (96. 6*96, 3)
```

### Addition and removal of axes

You can write `1` to create new axis of length 1. 
There is also a synonym `()` that does exactly the same

It is exactly what `numpy.exapand_axis()` and `torch.unsqueeze()` does.

```python
## both operations does the same as "numpy.expand_dims()" or "torch.unsqueeze()"
u = rearrange(images, "b h w c -> b 1 h w 1 c")
v = rearrange(images, "b h w c -> b () h w () c")

u.shape         ## (6, 1, 96, 96, 1, 3)
v.shape         ## (6, 1, 96, 96, 1, 3)
```

The `numpy.squeeze()` operation is also facilitated by `rearrange()` as usual.
```python
u = rearrange(images, "b h w c -> b 1 h w 1 c")         ## torch.unsqueeze()
v = rearrange(u, "b 1 h w 1 c -> b h w c")              ## torch.unsqueeze()

v.shape                                                 ## (6, 96, 96, 3)
```

???+ danger "An example usage"
    Compute max in each image individually and then show a difference
    ```python
    x = reduce(images, "b h w c -> b () () c", max)
    x -= images
    y = rearrange(x, "b h w c -> h (b w) c")
    y.shape                                             ## (96, 6*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_17.png)
    </figure>

### Meet "repeat": Repeating elements

This is the third operation in `einops` library

:dart: Repeat **along a new axis**. The new axis can be placed anywhere.
```python
u = repeat(images[0], "h w c -> h new_axis w c", new_axis=5)
u.shape         ## (96, 5, 96, 3)

## -- OR -- a shortcut

v = repeat(images[0], "h w c -> h 5 w c")   ## repeats 5 times in the new axis.
v.shape         ## (96, 5, 96, 3)
```

:dart: Repat along **an existing axis**
```python
## repeats the width 3 times
u = repeat(images[0], "h w c -> h (repeat w) c", repeat=3)
u.shape         ## (96, 3*96, 3)
```
<figure markdown class="card">
![](../../../assets/blogs/deep_learning/einops/images_18.png)
</figure>

:dart: Repeat along **multiple existing axes**
```python
u = repeat(images[0], "h w c -> (2 h) (2 w) c")
u.shape         ## (2*96, 2*96, 3)
```
<figure markdown class="card">
![](../../../assets/blogs/deep_learning/einops/images_19.png)
</figure>

:dart: Order of axes matter as usual. You can repeat each pixel 3 times by changing the order of axes in repeat
```python
## repeat the pixels along the width dim. 3 times
u = repeat(images[0], "h w c -> h (w repeat) c", repeat=3)
u.shape         ## (96, 96*3, 3)
```
<figure markdown class="card">
![](../../../assets/blogs/deep_learning/einops/images_20.png)
</figure>

:man_raising_hand: NOTE: The `repeat` operation covers `numpy.tile`, `numpy.repeat` and much more.


### reduce v/s repeat

`reduce` and `repeat` are opposite of each other. 

1. `reduce`: reduces amount of elements
2. `repeat`: increases the number of elements.

???+ danger "An example of `reduce` v/s `repeat`"
    In this example each image is repeated first then reduced over the `new_axis`
    to get back the original tensor.
    ```python
    repeated = repeat(images, "b h w c -> b h new_axis w c", new_axis=2)
    reduced = reduce(repeated, "b h new_axis w c -> b h w c", "min")

    repeated.shape                                  ## (6, 96, 2, 96, 3)
    reduced.shape                                   ## (6, 96, 96, 3)

    assert numpy.array_equal(images, reduced)       ## True
    ```
    **Notice that the operation pattern in `reduce` and `repeat` are reverse of each other.**
    i.e. 

    in `repeat` its `"b h w c -> b h new_axis w c"` but
    
    in `reduce` its `"b h new_axis w c -> b h w c"`


### Some more examples

???+ quote "Interwaving pixels of different pictures"
    All letters can be observed in the final image
    ```python
    u = rearrange(images, "(b1 b2) h w c -> (h b1) (w b2) c", b1=2)
    u.shape             ## (2*96, 3*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_21.png)
    </figure>

???+ done "Interweaving along vertical for couple of images"
    ```python
    u = rearrange(images, "(b1 b2) h w c -> (h b1) (b2 w) c", b1=2)
    u.shape             ## (96*2, 3*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_22.png)
    </figure>

???+ quote "Interweaving lines for couple of images"
    ```python
    u = reduce(images, "(b1 b2) h w c -> h (b2 w) c", "max", b1=2)
    u.shape             ## (96, 3*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_23.png)
    </figure>

???+ done "Decomposing color into different axes"
    Here we decompose color dimension into different axes. We also downsample the image.
    ```python
    u = reduce(images, "b (h 2) (w 2) c -> (c h) (b w)", "mean")
    u.shape             ## (3*48, 6*48)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_24.png)
    </figure>

???+ quote "Disproportionate resize"
    ```python
    u = reduce(images, "b (h 3) (w 4) c -> (h) (b w)", "mean")
    u.shape             ## (96/3, 6*96/4)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_25.png)
    </figure>

???+ done "Split & Reduce"
    Split each image into two halves and compute the mean of the two halves.
    ```python
    u = reduce(images, "b (h1 h2) w c -> h2 (b w)", "mean", h1=2)
    u.shape             ## (96/2, 6*96)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_26.png)
    </figure>

???+ quote "Split and Transpose"
    Split into small patches and transpose each patch.
    ```python
    ## splitting each image into 96/8 * 96/8 = 12*12 = 144 patches
    ## each patch is of shape (8, 8)
    u = rearrange(images, "b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c", h2=8, w2=8)
    u.shape             ## (12*8, 6*12*8, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_27.png)
    </figure>

???+ done "Another Split & Transpose"
    This is crazy
    ```python
    u = rearrange(images,
                  "b (h1 h2 h3) (w1 w2 w3) c -> (h1 w2 h3) (b w1 h2 w3) c",
                  h2=2, h3=2, w2=2, w3=2)
    u.shape             ## (96/(2*2)*2*2, 6*96/(2*2)*2*2, c) = (96, 6*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_28.png)
    </figure>

???+ quote "Yet another Split & Transpose"
    This is crazy crazy....
    ```python
    u = rearrange(images,
                  "(b1 b2) (h1 h2) (w1 w2) c -> (h1 b1 h2) (w1 b2 w2) c",
                  h1=3, w1=3, b2=3)
    u.shape             ## (3*(6/3)*(96/3), 3*3*(96/3), 3) = (192, 288, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_29.png)
    </figure>

???+ danger "Arbitrarily Complicated Pattern"
    ```python
    u = reduce(images,
               "(b1 b2) (h1 h2 h3) (w1 w2 w3) c -> (h1 w1 h3) (b1 w2 h2 w3 b2) c", 
               "mean",
               w1=2, w3=2, h2=2, h3=2, b2=2)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_30.png)
    </figure>

???+ quote "Subtract background & Normalize"
    Subtract background in each image individually and normalize.

    ** :dart: NOTE: Pay attention to `()` -- this is a composition of `0` axis
    (a dummy axis with 1 element)**

    ```python
    u = reduce(images, "b h w c -> b () () c", "max")   ## finding per-image per-channel max
    u -= images                                         ## subtracting
    u /= reduce(u, "b h w c -> b () () c", "max")       ## NORMALIZATION
    u = rearrange(u, "b h w c -> h (b w) c")

    u.shape                                             ## (96, 6*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_31.png)
    </figure>

???+ danger "PIXELATE"
    First **downscale** by averaging then **upscale** by using the same pattern.
    ```python
    ## downscale using "mean" kernel of size (6, 8)
    downscaled = reduce(images, "b (h h2) (w w2) c -> b h w c", "mean", h2=6, w2=8)
    upscaled = repeat(downscaled, "b h w c -> b (h h2) (w w2) c", h2=6, w2=8)
    v = rearrange(upscaled, "b h w c -> h (b w) c")

    downscaled.shape            ## (6, 96/6, 96/8, 3)
    upscaled.shape              ## (6, (96/6)*6, (96/8)*8, 3) = (6, 96, 96, 3)
    v.shape                     ## (96, 6*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_32.png)
    </figure>

???+ quote "ROTATE"
    ```python
    u = rearrange(images, "b h w c -> w (b h) c")       ## rotation of (width <-> height) 
    u.shape             ## (96, 6*96, 3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_33.png)
    </figure>

???+ quote "Another Example"
    Let's bring the `channel` dimension as part of the `width` axis.

    Also, at the same time **downsample** the `width` axis by 2x
    ```python
    u = reduce(images, 
               "b (h h2) (w w2) c -> (h w2) (b w c)", 
               "mean", 
               h2=3, w2=3)
    ```
    <figure markdown class="card">
    ![](../../../assets/blogs/deep_learning/einops/images_34.png)
    </figure>


