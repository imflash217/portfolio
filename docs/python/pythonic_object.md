Lets start by introducing a `Vector` class
```python
from array import array
import math

class Vector2D:
    typecode = "d"
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))

    def __repr__(self):
        class_name = type(self).__name__
        return "{}({!r}, {!r})".format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) + bytes(array(self.typecode, self)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))
```

## An alternative constructor

Since in above example, we **export** a `Vector2D` as bytes; 
we also need a method to **imports** a `vector2D` from binary sequence.

Looking at the standard library for inspiration, we see that `array.array`
has a class method `frombytes()`. Lets adopt this nomenclature in our `Vector2D` class.

```python
## adding a import method from a binary sequence
## inside Vector2D class definition

@classmethod
def frombytes(cls, octets):
    typecode = chr(octets[0])
    memv = memoryview(octets[1:]).cast(typecode)
    return cls(*memv)
```

## @classmethod v/s @staticmethod

**`@classmethod`**:

1. It is used to define a method that operates on **class**; not on **instances**.
2. It receives the **class** itself as the 1st argument (eg. `cls` in above code) 
instead of an instance (eg. `self`)
3. Most commonly used as **alternate constructors** (eg. `frombytes` in above code)
Note: (in the above `frombyytes()`) how the last line in `frombytes()` uses `cls` argument by invoking 
it to build a new instance (`cls(*memv)`)
4. By convention the 1st argument of the `@classmethod` should be named `cls` 
(but Python does not care about the name though)

**`@staticmethod`**

A static method is just like a plain old **function** that happens to live inside the body of a **class**
instead of being defind outside the class.
It does not have access to internal state variables of the class or the instance.
It is kept inside the class definition to provide easy access to related functions/method,
so the user have access to all necessary method for a class within itself instead of finding it elsewhere.

An example:
```python
class Demo:
    @classmethod
    def klassmeth(*args):
        return *args

    @staticmethod
    def statmeth(*args):
        return *args
```
```python
## no matter how it is invoked, Demo.klassmethod always receives Demo class as its 1st argument
Demo.klassmeth()            ## (<class "__main__.Demo">,)
Demo.klassmeth("spam")      ## (<class "__main__.Demo">, "spam")

Demo.statmeth()             ## ()
Demo.statmeth("spam")       ## ("spam")
```

## Formatted displays

1. `int` type supports `b` (for base=2 integers) and `x` (for base=16 integers).
2. `float` type implements `f` (for fixed-points) and `%` (for a percentage display).

```python
format(42, "b")         ## "101010"
format(42, "x")         ## "2a"

format(2/3, 'f')        ## 0.666667
format(2/3, ".3f")      ## 0.667

format(2/3, "%")        ## 66.666667%
format(2/3, ".2%")      ## 66.67%
```
 The Format Specifier Mini Language is extensible because each class gets to interpret
 the `format_spec` argument as it likes. 

 ```python
from datetime import datetime
now = datetime.now()
format(now, "%H:%M:%S")             ## "18:49:05"
print("Its now {:%I:%M %p}")        ## "Its now 06:49 PM"
```
If a class has no `__format__()` method, the method inherited from `object` return `str(my_object)`.

