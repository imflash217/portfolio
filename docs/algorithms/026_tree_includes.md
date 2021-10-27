<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

<!-- ######################################################################################################### -->

## Problem

???+ danger "Problem"

    Write a function, tree_includes, that takes in the root of a binary tree and a target value. 
    The function should return a boolean indicating whether or not the value is contained in the tree.
    
    ```python
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    e = Node("e")
    f = Node("f")

    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f

    #      a
    #    /   \
    #   b     c
    #  / \     \
    # d   e     f

    tree_includes(a, "e") # -> True
    ```

## Solution

???+ done "Solution"
    ```python
    # class Node:
    #   def __init__(self, val):
    #     self.val = val
    #     self.left = None
    #     self.right = None

    def tree_includes(root, target):
      
      ## base case of leaf-nodes
      if not root: return False
      
      ## success
      if root.val == target: return True
      
      ## else search in the children tree
      return tree_includes(root.left, target) or tree_includes(root.right, target)

    ```

## Discussion

???+ quote "Discussion"
    `...`

