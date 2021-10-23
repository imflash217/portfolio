<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

## `Strategy` Pattern

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

## `Function-oriented` Strategy Pattern

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


