<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Part-1

## Welocome to `einops`

1. We don't write 
    `y = x.transpose(0,2,3,1)`
2. We write comprehensible code
    `y = einops.rearrange(x, "b c h w -> b h w c")`
3. `einops` supports widely used tensor packages viz. 
    `numpy`, `pytorch`, `tensorflow`, `chainer`, `gluon`
    and **extends** them.

