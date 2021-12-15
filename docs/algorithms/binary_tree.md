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


## 30: Root-to-Leaf path w/ MAX sum

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

???+ done "Solution"
    ```python hl_lines="11-17 21"
    import math

    class Node:
        def __init__(self, value):
            self.val = value
            self.left = None
            self.right = None

    ## RECURSIVE Solution
    def max_path_sum(root):
        ## base case (None node)
        if root is None:
            return -math.inf
        
        ## base case (leaf node)
        if root.left == None and root.right == None:
            return root.val

        left_sum = max_path_sum(root.left)
        right_sum = max_path_sum(root.right)
        return root.val + max(left_sum, right_sum)
    ```

## 31: Tree Path Finder

???+ danger "Problem"
    Write a function **`path_finder`** that takes in the *root* of a BT and a *target value*
    The function should return an array representing the path to teh target value.
    If the target value is not found, then return `None`.
    
    Assumption: Every node in teh tree contains unique value.
    
    Sample example:

    ```python
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    e = Node("e")
    f = Node("f")
    g = Node("g")
    h = Node("h")
    
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f
    e.left = g
    f.right = h
    
    #      a
    #    /   \
    #   b     c
    #  / \     \
    # d   e     f
    #    /       \
    #   g         h
    
    path_finder(a, "h") # -> ['a', 'c', 'f', 'h']
    ```

???+ done "Solution"
    ```python hl_lines="36"
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def _path_finder(root, target):
        ## base case (None node)
        if root is None:
            return None

        ## base case (node found)
        if root.val == target:
            return [root.val]

        ## recuse over the left child
        left_path = _path_finder(root.left, target)
        if left_path is not None:
            ## target found in the left child subtree
            left_path.append(root.val)
            return left_path
        
        ## recurse over the right child
        right_path = _path_finder(root.right, target)
        if right_path is not None:
            ## target found in the left child subtree
            right_path.append(root.val)
            return right_path

        return None     ## edge case

    def path_finder(root, target):
        path = _path_finder(root, target)
        if path is None:
            return None
        return path[::-1]       ## return the path in reverse order (from root to target)
    ```

## 32: Tree Value Count

???+ danger "Problem"
    Write a function, tree_value_count, that takes in the root of a binary tree and a target value. 
    The function should return the number of times that the target occurs in the tree.
    
    ```python
    a = Node(12)
    b = Node(6)
    c = Node(6)
    d = Node(4)
    e = Node(6)
    f = Node(12)
    
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f
    
    #      12
    #    /   \
    #   6     6
    #  / \     \
    # 4   6     12
    
    tree_value_count(a,  6) # -> 3
    ```

???+ done "Solution"
    ```python
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    ## RECURSIVE DFS
    ## time = O(n)
    ## space = O(n)

    def tree_value_count(root, target):
        ## base case (None node)
        if root is None:
            return 0

        left_count = tree_value_count(root.left, target)
        right_count = tree_value_count(root.right, target)
        if root.val == target:
            return 1 + left_count + right_count
        return left_count + right_count
    ```

## 33: Height of a BT

???+ danger "Problem"
    Write a function, `how_high()`, that takes in the root of a binary tree. 
    The function should return a number representing the height of the tree.

    The height of a binary tree is defined as the maximal number of edges from the root node to any leaf node.

    **If the tree is empty, return -1.**

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
    
    how_high(a) # -> 3
    ```

???+ done "Solution"
    ```python
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    ## RECURSIVE approach
    def how_high(root):
        ## base case (None node)
        if root is None:
            return -1       ## see definition of height of a BT above
        
        left_height = how_high(root.left)
        right_height = how_high(root.right)
        return 1 + max(left_height, right_height)
    ```

## 34: Botton Right Value

???+ danger "Problem"
    Write a function, `bottom_right_value()`, that takes in the root of a binary tree. 
    The function should return the right-most value in the bottom-most level of the tree.

    You may assume that the input tree is non-empty.

    ```python
    a = Node(-1)
    b = Node(-6)
    c = Node(-5)
    d = Node(-3)
    e = Node(-4)
    f = Node(-13)
    g = Node(-2)
    h = Node(6)
    
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    c.right = f
    e.left = g
    e.right = h
    
    #        -1
    #      /   \
    #    -6    -5
    #   /  \     \
    # -3   -4   -13
    #     / \       
    #    -2  6
    
    bottom_right_value(a) # -> 6
    ```

???+ done "Solution"
    ```python hl_lines="1 4 14"
    from collections import deque
    
    def bottom_right_value(root):
        queue = deque([root])
        current = None

        while queue:
            current = queue.popleft()
            if current.left is not None:
                queue.append(current.left)
            if current.right is not None:
                queue.append(current.right)

        return current.val
    ```

## 35: All tree paths

Write a function, **`all_tree_paths()`**, that takes in the root of a binary tree. 
The function should return a 2-Dimensional list where each subarray represents 
a root-to-leaf path in the tree.

The order within an individual path must start at the root and end at the leaf, 
but the relative order among paths in the outer list does not matter.

You may assume that the input tree is non-empty.

```python
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g = Node('g')
h = Node('h')
i = Node('i')

a.left = b
a.right = c
b.left = d
b.right = e
c.right = f
e.left = g
e.right = h
f.left = i

#         a
#      /    \
#     b      c
#   /  \      \
#  d    e      f
#      / \    /   
#     g  h   i 

all_tree_paths(a) # ->
# [ 
#   [ 'a', 'b', 'd' ], 
#   [ 'a', 'b', 'e', 'g' ], 
#   [ 'a', 'b', 'e', 'h' ], 
#   [ 'a', 'c', 'f', 'i' ] 
# ] 
```

???+ done "Solution"
    
