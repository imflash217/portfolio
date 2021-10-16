<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

<!-- ######################################################################################################### -->

## `Problem`

???+ danger "Problem"
    Write a function, `tree_sum`, that takes in the root of a **binary tree** that contains number values. 
    The function should **return the total sum of all values** in the tree.
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
    # 4   -2     1

    tree_sum(a) # -> 21
    ```

## `Solution`
???+ done "Solution"
    ```python
    # class Node:
    #   def __init__(self, val):
    #     self.val = val
    #     self.left = None
    #     self.right = None

    def tree_sum(root):
      if not root: return 0                     ## base-case of leaf nodes
      left_sum = tree_sum(root.left)            ## sum of left subtree
      right_sum = tree_sum(root.right)          ## sum of right subtree
      return root.val + left_sum + right_sum
    ```


## `Discussion`

???+ quote "Discussion"
    `...`

