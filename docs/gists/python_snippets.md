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
