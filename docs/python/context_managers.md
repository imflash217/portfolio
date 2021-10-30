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

