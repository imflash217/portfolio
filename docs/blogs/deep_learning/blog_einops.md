<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Part-1

## Welocome to `einops`

We don't write 
`y = x.transpose(0,2,3,1)`

We write comprehensible code
`y = einops.rearrange(x, "b c h w -> b h w c")`
