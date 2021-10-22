<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

## Part-1

### Welocome to `einops`

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

### Preparations:
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
<figure markdown>
![...](../../../assets/blogs/deep_learning/einops/images_0.png)
</figure>

```python
images[1]
```
<figure markdown>
![...](../../../assets/blogs/deep_learning/einops/images_1.png)
</figure>

**We will use three opeartions:** `rearrange`, `reduce`, `repeat`
```python
from einops import rearrange, reduce, repeat
```

???+ done "rearrange"
    As its name suggests; it rearranges elements. Below, we swap `height` and `width`.

    In other words, below we **transpose** first two axes/dimensions.
    ```python
    rearrange(images[0], "h w c -> w h c")
    ```
    <figure markdown> 
        ![images_2](../../../assets/blogs/deep_learning/einops/images_2.png)
        <figcaption>`rearrange()`</figcaption>
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
    <figure markdown> 
        ![images_3](../../../assets/blogs/deep_learning/einops/images_3.png)
    </figure>


???+ danger "composition using `rearrange()`: width"
    `einops` allows seamlessly composing `batch` and `width` to a `new width` dimension.

    Below we just rendered all images in the 4D tensor by collapsing it to a 3D tensor.
    ```python
    rearrange(images, "b h w c -> h (b w) c")
    ```
    <figure markdown> 
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
    <figure markdown>
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
    <figure markdown>
    ![](../../../assets/blogs/deep_learning/einops/images_6.png)
    </figure>
