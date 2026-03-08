---
title: "Sorting Algorithms"
tags:
  - algorithms
  - sorting
summary: "Insertion sort, binary insertion sort, merge sort, and heap sort with pseudocode and complexity analysis."
---

# Sorting Algorithms

## Insertion Sort

### Vanilla Insertion Sort

```
for i = 1, 2, 3, ..., n{
    insert A[i] into sorted array A[0:i-1]
    by "pairwise swaps" down to the correct position
}
```

This above version has **$\theta(n)$** steps and each step has **$\theta(n)$** comparisons. So this version of the algorithm is **$\theta(n^2)$** runtime complexity.

### Binary Insertion Sort

This improved version is slightly improved by using **Binary Search** while searching for the position to place the key `A[i]` in the sorted part of the array (i.e. `A[0:i-1]`)

```
for i = 1, 2, 3, ..., n{
    insert A[i] into sorted array A[0:i-1]
    by "Binary Search" down to the correct position
}
```

This above version has **$\theta(n)$** steps and each step has **$\theta(\log n)$** comparisons. So this version of the algorithm has **$\theta(n \log n)$** comparisons; but **$\theta(n^2)$** swaps. So, overall complexity is still **$\theta(n^2)$**

## Merge Sort

```
merge_sort(A):
    if len(A) < 2: return A
    else:
        L = merge_sort(A[0:n/2])
        R = merge_sort(A[n/2:n])
        return merge(L, R)
```

**Time Complexity**: $T(n) = c_1 + 2T(n/2) + cn$

By Master's Theorem: $T(n) = \theta(n \log n)$

**Space Complexity**: $\theta(n)$ auxiliary space

## Heap Sort

- The HEAP is an implementation of the abstract data-type called **priority queue**
- An array visualized as a nearly complete Binary-Tree

### Heap properties:
- `parent(i) = i / 2`
- `left_child(i) = 2 * i`
- `right_child(i) = 2 * i + 1`

### Max-Heap Property:
For every node `i` (other than the root): `A[parent(i)] >= A[i]`

### Build Max Heap
```python
def build_max_heap(A):
    n = len(A)
    for i in range(n//2, -1, -1):
        max_heapify(A, i)

def max_heapify(A, i):
    l = 2*i + 1
    r = 2*i + 2
    largest = i
    if l < len(A) and A[l] > A[largest]:
        largest = l
    if r < len(A) and A[r] > A[largest]:
        largest = r
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        max_heapify(A, largest)
```