# Context Managers

A context manager is a Python object that provides extra contextual information to an action.
This extra contextual information takes the form of running a **callable** upon initiating the 
context using the `with` statement; as well as running a callable upon completing all the code 
inside the `with` block. 

For eg:

```python
with open("file.txt") as f:
    contents = f.read()
```

Anyone familiar with this pattern knows that invoking `open` in this fashion ensures that 
`f`'s `close()` will be called at some point. 

There are two ways to implement this functionality ourselves:

1. using `class`
2. using `@contextmanager` decorator

### Ctx Manager using CLASS

```python
class CustomOpen:
    def __init__(self, filename):
        self.file = open(filename)

    def __enter__(self):
        return self.file

    def __exit__(self, ctx_type, ctx_vale, ctx_traceback):
        self.file.close()

###################################################################

with CustomOPen("file.txt") as f:
    contents = f.read()

```

This is just a regular class with two extra methods `__enter__()` and `__exit__()`.
Implementation of `__enter__` and `__exit__` are essential for its usage in `with` statement.
Following are the three steps of functionality of `with` statement:

1. Firstly, `CustomOpen` is initantiated
2. Then its `__enter__()` method is called and whatever `__enter__()` returns is 
assigned to `f` in `as f` part of the statement.
3. When the **contents of the `with` block** is finished executing, 
then, `__exit__()` method is called.

### Ctx Managers using GENERATORS
```python
## implementing a smilar context manager as above 
## uisng a decorator

from contextlib import contextmanager

@contextmanager
def custom_open(filename):
    f = open(filename)
    try:
        yield f
    finally:
        f.close()

#########################################################

with custom_open("file.txt") as f:
    contents = f.read()

```

This works in exactly the same manner as the CLASS version.
The `custom_open` function executes until it reaches the `yield` statement.
The control was given back to the `with` statement which assigns whatever was
`yield`ed to `f` in the `as f` part  of the `with` statement.
The `finally` block is executed at the end of the `with` statement.
