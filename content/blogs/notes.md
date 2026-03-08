---
title: "Notes"
tags:
  - tips
  - deep-learning
summary: "Practical tips on LR schedulers, optimizers, and training tricks for deep learning models."
---

# The Preface of the key technological stuffs here

## Tips & Tricks

### LR Scheduler
- [x] Similar to the `learning rate`, the `lr-scheduler` to apply depends on the
    classifier & the model.
- [x] For image classifiers and **`SGD` optimizer**, the **`Multi-Step LR Scheduler`**
    is shown to be a good choice.
- [x] Models trained with **`Adam`** commonly use a smooth exponential-decay
    in the `lr` or a cosine-like scheduler.
- [x] For TRANSFORMERS:
    - 🚨 Remember to use a **`learning rate WARMUP`**
    - 🚨 The `cosine-scheduler` is often used for decaying the `lr`
        afterwards (but can also be replaced by `exponential decay`)

### Regularization
- [x] Regularization is important in networks when we see a significantly higher
    **training** performance than **test** performance.
