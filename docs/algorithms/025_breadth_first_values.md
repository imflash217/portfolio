<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

<!-- ######################################################################################################### -->
## `Problem`

???+ danger "Breadth First Values"
    Write a function, `breadth_first_values`, that takes in the root of a binary tree. 
    The function should return a list containing all values of the tree in **breadth-first order**.
    
    ```python
        a = Node('a')
    b = Node('b')
    c = Node('c')
    d = Node('d')
    e = Node('e')
    x = Node('x')

    a.right = b
    b.left = c
    c.left = x
    c.right = d
    d.right = e

    #      a
    #       \
    #        b
    #       /
    #      c
    #    /  \
    #   x    d
    #         \
    #          e

    breadth_first_values(a) 
    #    -> ['a', 'b', 'c', 'x', 'd', 'e']
    ```

## `Solution`

???+ done "Solution"
    ```python
    # class Node:
    #   def __init__(self, val):
    #     self.val = val
    #     self.left = None
    #     self.right = None

    def breadth_first_values(root):
      ## base case for empty tree
      if root is None:
        return []
      
      ## using double-ended queue struct
      from collections import deque
      
      ## step-1: add the root into the queue
      queue = deque([root])
      visited = []
      
      while queue:
        ## grabbing current visited node
        current_node = queue.popleft()
        
        ## put that node in the visited list
        visited.append(current_node.val)
        
        ## and add their children (if present) into the queue
        if current_node.left:
          queue.append(current_node.left)
        if current_node.right:
          queue.append(current_node.right)
        
      return visited
    ```

## `Discussion`

???+ quote "Discussion"
    ...
