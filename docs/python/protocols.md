
In the context of Object Oriented Programming, 
a **protocol** is an _informal interface that is defined only in the documentation, not in code_.

For eg., the **sequence protocol** in Python entails just the `__len__()` and `__getitem__()` methods.
Any class `Spam` that uses those methods can be used as a **sequence**.
Whether `Spam` is a subclass of this or that is irrelevant; 
all that matters is that it provides the necessary methods

```python
import collections
Card = collections.namedtuple("Card", ["rank", "suit"])

class FrenchDeck:
    ranks = [str(i) for i in range(2, 11)] + str("JQKA")
    suits = "spades, diamonds, clubs, hearts".split(", ")

    def __init__(self):
        self._cards = [Card(rank, suit) for rank in ranks
                                        for suit in suits]

    def __len__(self):
        ## necessary for usage as a SEQUENCE
        return len(self._cards)

    def __getitem__(self, idx):
        ## necessary for usage as a SEQUENCE
        return self._cards[idx]
```

Because **protocols** are informal and un-enforced, you can get away with just 
implementing the part of the protocol that is necessary for your usage.

For example, to support only **iteration** we just need to implement the `__getitem__()` method;
we don't need to implement `__len__()`.


