## 25: Depth First values

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


## 26: Breadth First Values

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



## 27: Tree Includes

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


## 28: Tree Sum

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


## 29: Binary Tree Min Value

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


## 30: Root-to-Leaf path with MAX sum

???+ danger "Problem"
    Write a function **`max_path_sum()`** that takes in the root of a Binary Tree
    that conatins number values. The functions should return the maximum sum of any 
    root-to-leaf path in the tree.
    
    Assumption: The tree is non-empty
    
    An example of the scenario is shown below:
    
    ```python
    a = Node(-1)
    b = Node(-6)
    c = Node(-5)
    d = Node(-3)
    e = Node(0)
    f = Node(-13)
    g = Node(-1)
    h = Node(-2)
    
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f
    e.left = g
    f.right = h
    
    #        -1
    #      /   \
    #    -6    -5
    #   /  \     \
    # -3   0    -13
    #     /       \
    #    -1       -2
    
    max_path_sum(a) # -> -8
    ```



