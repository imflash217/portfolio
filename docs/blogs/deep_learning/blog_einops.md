<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Part-1

## Welocome to `einops`

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

## What's in this tutorial?

1. **Fundamentals**: reordering, composition, and decomposition of tensors.
2. **Operations**: `rearrange`, `reduce`, `repeat`
3. How much can you do with a **single** operation?
