<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

<!-- ######################################################################################################### -->
## Problem

???+ danger "Depth First Values"
    Write a function, `depth_first_values`, that takes in the root of a binary tree. 
    The function should return a list containing all values of the tree in **depth-first order**.
    
    ```python
    a = Node('a')
    b = Node('b')
    c = Node('c')
    d = Node('d')
    e = Node('e')
    f = Node('f')
    g = Node('g')
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f
    e.left = g

    #      a
    #    /   \
    #   b     c
    #  / \     \
    # d   e     f
    #    /
    #   g

    depth_first_values(a)
    #   -> ['a', 'b', 'd', 'e', 'g', 'c', 'f']

    ```

???+ done "Solution"
    ```python
    # class Node:
    #   def __init__(self, val):
    #     self.val = val
    #     self.left = None
    #     self.right = None

    def depth_first_values(root):
      """recursive solution Depth First Traversal"""
      
      ## base case
      if root is None:
        return []
      
      left_values = depth_first_values(root.left)
      right_values = depth_first_values(root.right)
      
      ## in DFS, the visiting order is
      ## root, left-child, right-child
      return [root.val, *left_values, *right_values]

    ```

