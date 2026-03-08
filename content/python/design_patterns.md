---
title: "Design Patterns"
tags:
  - python
  - design-patterns
summary: "Strategy pattern in Python using ABC and namedtuples for flexible discount policies."
---

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
        return f"<Order total: {self.total():.2f} due: {self.due():.2f}>"
```

### Abstract Strategy

```python
class Promotion(ABC):
    """Abstract Base Class for implementing promotions"""

    @abstractmethod
    def discount(self, order):
        """return discount as +ve amount"""
        pass

class FidelityPromotion(Promotion):
    """5% off for customers with 1000+ fidelity points"""
    def discount(self, order):
        return order.total() * 0.05 if order.customer.fidelity >= 1000 else 0

class BulkItemPromotion(Promotion):
    """10% discount for each item with 20+ units"""
    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * 0.1
        return discount

class LargeOrderPromotion(Promotion):
    """7% discount on orders with 10+ distinct items"""
    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * 0.07
        return 0
```