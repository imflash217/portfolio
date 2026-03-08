---
title: "Hash Tables"
tags:
  - algorithms
  - data-structures
  - hash-tables
author: "Vinay Kumar"
date: 2021-01-29
summary: "Deep dive into hash tables  -  hashing, collision resolution via chaining and open addressing, and hash functions."
---

# Hash Tables
`Author: Vinay Kumar (@imflash217) | Date: 29/January/2021`

## Definition

**Hash Table** is a data structure which stores data in an **associative manner** (i.e. in a (key, value) pair).

- In a hash table, the data is stored in an array format where each data-value has its own unique index-value. Due to this feature, the access to data becomes very fast if we know the desired index-value; irrespective of the size of the data.
- Hash Table uses an array as a storage medium and uses **hashing** to generate the index where an element is to be inserted or to be located from.

## Hashing

**Hashing** is a technique to convert a range of key values into a range of indexes of an array.

### Direct-Address Tables (DAT)

DAT works well when the universe `U` of keys is small. But when `U` is very large then it is IMPRACTICAL to have all `|U|` slots.

### Hash Functions

A good hash function satisfies the assumption of **simple uniform hashing**: each key is equally likely to hash to any of the `m` slots.

#### Division Method:
$$h(k) = k \mod m$$

#### Multiplication Method:
$$h(k) = \lfloor m(k \cdot A \mod 1) \rfloor$$
where $0 < A < 1$. Knuth suggests $A \approx \frac{\sqrt{5} - 1}{2} = 0.6180339887...$

### Collision Resolution

#### Chaining
- Each slot of the hash-table holds a linked list of all the elements that hash to the same slot.
- **Expected Time**: $O(1 + \alpha)$ where $\alpha = n/m$ is the load factor

#### Open Addressing
- All elements are stored in the hash-table itself.
- Each slot has either an element or `NIL`.
- **Linear Probing**: $h(k, i) = (h'(k) + i) \mod m$
- **Quadratic Probing**: $h(k, i) = (h'(k) + c_1 i + c_2 i^2) \mod m$
- **Double Hashing**: $h(k, i) = (h_1(k) + i \cdot h_2(k)) \mod m$