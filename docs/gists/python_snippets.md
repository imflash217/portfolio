<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Python Snippets
`Author: Vinay Kumar (@imflash217) | Update: 07/October/2021`

<!-- ######################################################################################################### -->

## `1: MappingProxyType`

???+ quote "Immutable Mappings"
    The mapping types provided by the standard library are all mutable; 
    but you may need to gurantee that a user cannot change a mapping by mistake.

    Since `Python 3.3` the `types` module provides a wrapper class `MappingProxyType` which,
    given a mapping returns a `mappingproxy` instance that is **read-only** but a **dynamic-view** 
    of the original mapping. 

    This means that the original mapping can be seen through `mappingproxy`
    but changes cannot be made through it.

    ```python
    from types import MappingProxyType

    d = {1:"A"}
    d_proxy = MappingProxyType(d)   ## creating a proxy for the original dict d
                                    ## d_proxy = {1:"A"}
    print(d_proxy[1])               ## "A"
    d_proxy[2] = "X"                ## TypeERROR. mappingproxy does not support item assignment

    d[2] = "B"                      ## OKAY. The original dictionary is still mutable

    print(d_proxy)                  ## The proxy has a dynamic view of the original dict. 
                                    ## So, it refelects the change
                                    ## {1:"A", 2:"B"}
    ```

<!-- ######################################################################################################### -->

## `2: Set operators`

???+ quote "Set Operators"
    ```markdown
    **operator:     method:                     desciption:**
                    `s.isdisjoint(z)`           `s` and `z` are disjoint (i.e. have no elements in common)
    `e in s`        `s.__contains__(e)`         element `e` is a subset of `s` set
    
    `s <= z`        `s.__le__(z)`               `s` is a **subset** of `z` set
                    `s.issubset(it)`            `s` is a **subset** of the set built from the iterable `it`
    `s < z`         `s.__lt__(z)`               `s` is a **PROPER-subset** of `z` set

    `s >= z`        `s.__ge__(z)`               `s` is a **superset** of `z` set
                    `s.issuperset(it)`          `s` is a **superset** of the set built from iterable `it`
    `s > z`         `s.__gt__(z)`               `s` is a **PROPER-superset** of the set `z`
    ```

## `3: set v/s frozenset`

???+ quote "Set v/s Frozenset"
    ```markdown
    **operator:     set:    frozenset:      description:**
    `s.add(e)`      ✅                      Add element `e` to set `s`
    `s.clear()`     ✅                      Remove all elements from set `s`
    `s.copy()`      ✅      ✅              Shallow copy of set/frozenset `s`
    `s.discard(e)`  ✅                      Remove element `e` from set `s` IF it is present
    `s.__iter__()`  ✅      ✅              Get iterator over set/frozenset `s`
    `s.__len__()`   ✅      ✅              `len(s)`
    `s.pop()`       ✅                      Remove and return an element from `s`; raising `keyError` if `s` is empty
    `s.remove(e)`   ✅                      Remive element `e` from set `s`; raise `KeyError` if `e not in s`
    ```
