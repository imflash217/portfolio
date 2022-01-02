# Dictionary Abstract Data Types

## Another Solution

- [ ] Can do better with **Hash Tables** in $O(1)$ expected time, $O(n+m)$ space; 
_where **m** is the table size_


## An example

- [ ] Let `keys` be the student ids of students registered in class **CLS201**; _eg._ `2022CS10110`.
- [ ] There are $100$ students in the class, so we create a hash-table of size _say_ 100.
- [ ] Hash function **`hash()`** is _say_, the last two digits of the _student-id_.
- [ ] So, now, `2022CS10110` goes to location `10` in the hash-table.
- [ ] 

## Multiply-Add-Divide (MAD)

This technique is used to create pseudo-unique **hash values**.

> :rotating_light: It is also used in **pseudo random number generators**

The method is as follows:

$$ \text{hash}(k) = |a\cdot k + b|\ \text{mod}\ N$$


## Collisions

Methods to resolve **collisions**

1. Chaining
2. Open Addressing:
    1. Linear Probing
    2. Double Hashing
