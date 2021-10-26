## Decorators 101

** A decorator is a `callable` that takes another function as argument (the decorated func.)**
The decorator may perform some processing with the decorated function, 
and return it or replaces it with another function or `callable` object.

Both these code snippet shas the same effect:
```python
@decorate
def target():
    print("running target()")
```
```python
## this code snippet has the same effect as the above one (using @decorate decorator)
def target():
    print("running target()")

target = decorate(target)
```

