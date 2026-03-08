---
title: "Loss Functions"
tags:
  - deep-learning
  - loss-functions
summary: "Cross entropy loss explained with NumPy implementation for multiclass classification tasks."
---

# Loss Functions

## Cross Entropy

Cross Entropy is usually used in **multiclass classification** tasks.

> **`Pareto Optimization`**: An area of *multiple* criteria decision making that is concerned with mathematical optimization problems involving more than one objective functions to be optimized simultaneously.

### Cross Entropy using Numpy

```python
import numpy as np

def cross_entropy(preds, labels):
    xentropy = 0
    for i in range(len(preds)):
        xentropy -= preds[i] * np.log(labels[i])    # NOTE the `-=` instead of `+=`
    return xentropy
```