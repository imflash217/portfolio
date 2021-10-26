<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

## STRATEGY Pattern

```python
from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple("Customer", "name fidelity")

class LineItem:
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity

class Order:
    """This is the CONTEXT part of the Strategy-Pattern"""
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, "__total"):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        fmt = "< Order total = {:.2f}; DUE = {:.2f} >"
        return fmt.format(self.total(), self.due())

class Promotion(ABC):
    """
    The STRATEGY part of the Strategy-pattern
    An Abstract Base Class
    """
    @abstractmethod
    def discount(self, order):
        """Return discount as a positive dollar amount"""

class FidelityPromot(Promotion):
    """
    First CONCRETE implementation of STRATEGY ABC

    5% disount for customer with 1000 or more fidelity points
    """
    def discount(self, order):
        return order.total() * 0.05 if order.customer.fidelity >= 1000 else 0

class BulkPromo(Promotion):
    """
    Second CONCRETE implementation of the Strategy-pattern

    10% discount for each line-item with 20 or more units
    """
    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * 0.1
        return discount

class LargeOrderPromo(Promotion):
    """
    Third CONCRETE implementation of the Strategy-pattern

    7% discount for orders with 10 or more distinct items
    """
    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * 0.07
        return 0
```

???+ quote "Example Usage"
    Sample usage of `Order` class with different promotions applied

    ```python
    joe = Customer("John Doe", 0)
    ann = Customer("Ann Smith", 1100)

    cart = [LineItem("banana", 4, 0.5),
            LineItem("apple", 10, 1.5),
            LineItem("watermelon", 5, 5.0)]
    ```
    ```python
    Order(joe, cart, FidelityPromo())       ## < Order total = 42.00; DUE = 42.00 >
    Order(ann, cart, FidelityPromo())       ## < Order total = 42.00; DUE = 39.90 >
    ```
    Few more example usage with differnt cart types
    ```python
    banana_cart = [LineItem("banana", 30, 0.5),
                   LineItem("apple", 10, 1.5)]

    Order(joe, banana_cart, BulkItemPromo())    ## < Order total = 30.00; DUE = 28.50 >
    ```
    ```python
    long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

    Order(joe, long_order, LargeOrderPromo())   ## < Order total = 10.00; DUE = 9.30 >
    Order(joe, cart, LargeOrderPromo())         ## < Order total = 42.00; DUE = 42.00 >
    ```

## Function-oriented STRATEGY Pattern

Each concrete implementation of the Strategy Pattern in above code is a `class`
with a single method `discount()`. Furthermore, the strategy instances have no state
(i.e. no instance attributes).

They look a lot like plain functions. 
So, below we re-write the **concrete implementations** of the Strategy Pattern as plain _function_.

```python
from collections import namedtuple

Customer = namedtuple("Customer", "name fidelity")

class LineItem:
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity

class Order:
    """The CONTEXT"""
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, "__total"):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        discount = 0
        if self.promotion:
            discount = self.promotion(self)
        return self.total() - discount

    def __repr__(self):
        fmt = "<Order total = {:.2f}; DUE = {:.2f}>"
        fmt.format(self.total(), self.due())

########################################################################################
## Redesign of the concrete-implementations of STRATEGY PATTERN as functions

def fidelity_promot(order):
    """5% discount for customers with >= 1000 fidelity points"""
    return order.total() * 0.05 if order.customer.fidelity >= 1000 else 0

def bulk_item_promo(order):
    """10% discount for each LineItem with >= 20 units in cart"""
    discount = 0
    for item in oder.cart:
        if item.quantity >= 20:
            discount += item.total() * 0.1
    return discount

def large_order_promo(order):
    """7% discount for orders with >= 10 distinct items"""
    distinct_items = set(item.product for item in order.cart)
    if len(distinct_items) >= 10:
        return order.total() * 0.07
    return 0

```

???+ quote "Example Usage"
    Smaple usage examples of `Order` class with promotion Strategy as **functions**
    ```python
    joe = Customer("John Doe", 0)
    ann = Customer("Ann Smith", 1100)

    cart = [LineItem("banana", 4, 0.5),
            LineItem("apple", 10, 1.5),
            LineItem("watermelon", 5, 5.0)]
    ```
    ```python
    Order(joe, cart, fidelity_promo)                ## < Order total = 42.00; DUE = 42.00 >
    Order(ann, cart, fidelity_promo)                ## < Order total = 42.00; DUE = 39.90 >
    ```

    Another Example

    ```python
    banana_cart = [LineItem("banana", 30, 0.5),
                   LineItem("apple", 10, 1.5)]
    Order(joe, banana_cart, bulk_item_promo)        ## < Order total = 30.00; DUE = 28.50 >
    ```
    
    Yet another Example

    ```python
    long_order = [LineItem(str(item_id), 1, 1.0) for item_id in range(10)]

    Order(joe, long_order, large_order_promo)       ## < Order total = 10.00; DUE = 9.30 >
    Order(joe, cart, large_order_promo)             ## < Order total = 42.00; DUE = 42.00 >
    ```

1. **STRATEGY** objects often make good **FLYWEIGHTS**
2. A **FLYWIGHT** is a _shared_ object that cane be use din multiple contexts simulatenously.
3. Sharing is encouraged to reduce the creation of a new **concrete** strategy object when the 
same strategy is applied over and over again in different contexts (i.e. with every new `Order` instance)
4. If the **strategies** have no internal state (often the case); 
then use plain old **functions** else adapt to use **class** version.
5. A **function** is more lightweight than an user-defined `class`
6. A plain **functions** is also a_shared_ object that can be used in multiple contexts simulateneously.


## Choosing the best Strategy

Given the same customers and carts from above examples; we now add additional tests.

```python
Order(joe, long_order, best_promo)          ## < Order total = 10.00; DUE = 9.30 >      ## case-1
Order(joe, banana_cart, best_promo)         ## < Order total = 30.00; DUE = 28.50 >     ## case-2
Order(ann, cart, best_promo)                ## < Order total = 42.00; DUE = 39.90 >     ## case-3
```

* case-1: `best_promo` selected the `large_order_promo` for customer `joe`
* case-2: `best_promo` selected the `bulk_item_promo` for customer `joe` (for ordering lots of bananas)
* case-3: `best_promo` selected the `fidelity_promo` for `ann`'s loyalty.

Below is the implementation of `best_promo`

```python
all_promos = [fidelity_promo, bulk_item_promo, large_order_promo]

def best_promo(order):
    """Selects the best discount avaailable. Only one discount applicable"""
    best_discount = max(promo(order) for promo in all_promos)
    return best_discount
```

## Finding Strategies in a module

```python
## Method-1
## using globals()
all_promos = [globals()[name] for name in globals()
              if name.endswith("_promo")
              and name != "best_promo"]

def best_promo(order):
    best_discount = max(promo(order) for order in all_promos)
    return best_discount
```

But a more flexible way to handle this is using inbuilt `inspect` module 
and storing all the promos functions in a file `promotions.py`.
This works regardless of the names given to promos.

```python
## Method-2
## using modules to store the promos separately
all_promos = [func for name, func in inspect.getmembers(promotions, inspect.isfunction)]

def best_promo(order):
    best_discount = max(promo(order) for promo in all_promos)
    return best_discount
```

Both the methods have pros & cons. Choose as you see fit.

> We could add more stringent tests to filter the functions, 
> by inspecting their arguments for instance.

> A more elegant solution would be use a **decorator**. (we will study this in later blogs)

## References
[^1]: https://github.com/gennad/Design-Patterns-in-Python
