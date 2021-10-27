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
v/s
```python
## this code snippet has the same effect as the above one (using @decorate decorator)
def target():
    print("running target()")

target = decorate(target)
```
Decorators are just syntactic sugar. 
We can always call a decorator like any regular callable, passing another function (as shown in 2nd snippet above).

1. Decorators have the power to replace the decorated-function with a different one.
2. Decorators are executed immediately when a module is loaded.

## When does Python executs decorators?

A key feature of decorators is that they run right after the decorated function is defined.
This happens usually at the `import` time (i.e. when the module is loaded).

???+ quote "An example"
    ```python
    ## registration.py module
    registry = []

    ```
