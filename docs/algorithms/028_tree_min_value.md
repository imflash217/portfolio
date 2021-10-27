<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->
<!-- ######################################################################################################### -->
## Problem

???+ danger "Minimum value in a binary tree"
    Write a function, tree_min_value, that takes in the root of a binary tree that contains number values. 
    The function should return the minimum value within the tree.

    You may assume that the input tree is non-empty.

    ```python
    a = Node(3)
    b = Node(11)
    c = Node(4)
    d = Node(4)
    e = Node(-2)
    f = Node(1)

    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f

    #       3
    #    /    \
    #   11     4
    #  / \      \
    #  4   -2     1
    tree_min_value(a) # -> -2

    ```

## Solution

???+ done "Solution"
    ```python
    # class Node:
    #   def __init__(self, val):
    #     self.val = val
    #     self.left = None
    #     self.right = None

    def tree_min_value(root):
      import math
      
      if not root: return math.inf                ## base case for leaf nodes's children
      left_min = tree_min_value(root.left)        ## minimum value in left subtree
      right_min = tree_min_value(root.right)      ## minimum value in right subtree
      return min(root.val, left_min, right_min)   ## return minimum of root, lef, & right subtrees

    ```

## Discussion

???+ quote "Discussion"
    `...`






