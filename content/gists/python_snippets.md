---
title: "Python Snippets"
tags:
  - python
  - snippets
summary: "MappingProxyType, walrus operator, __slots__  -  useful Python snippets and idioms."
---

# Python Snippets

## 1: MappingProxyType

### Immutable Mappings

The mapping types provided by the standard library are all mutable;
but you may need to guarantee that a user cannot change a mapping by mistake.

Since `Python 3.3` the `types` module provides a wrapper class `MappingProxyType` which,
given a mapping returns a `mappingproxy` instance that is **read-only** but a **dynamic-view**
of the original mapping.

This means that the original mapping can be seen through `mappingproxy`
but changes cannot be made through it.

```python
from types import MappingProxyType

d = {1:"A"}
d_proxy = MappingProxyType(d)   ## creating a proxy for the original dict d

print(d_proxy)       ## mappingproxy({1: 'A'})
print(d_proxy[1])    ## 'A'

## d_proxy[2] = "B"  ## THIS WILL RAISE TypeError

d[2] = "B"           ## changing the original dict `d`
print(d_proxy)       ## mappingproxy({1: 'A', 2: 'B'})  -- reflected in proxy!
```

## 2: Walrus Operator

The **walrus operator** `:=` (assignment expression) was introduced in Python 3.8.

```python
## Without walrus operator
results = []
for x in range(10):
    y = x ** 2
    if y > 20:
        results.append(y)

## With walrus operator
results = [y for x in range(10) if (y := x**2) > 20]
```

## 3: `__slots__`

By default, Python stores instance attributes in a per-instance `__dict__`.
When you have millions of instances, this overhead adds up. Using `__slots__`
tells Python to store instance attributes in a fixed-size tuple instead.

```python
class PointSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# This is much more memory efficient than:
class PointDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```