<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Python Snippets
`Author: Vinay Kumar (@imflash217) | Update: 07/October/2021`

<!-- ######################################################################################################### -->

## `1: MappingProxyType`

???+ quote "Immutabel Mappings"
    The mapping types provided by the standard library are all mutable; 
    but you may need to gurantee that a user cannot change a mapping by mistake.

    Since `Python 3.3` the `types` module provides a wrapper class `MappingProxyType` which,
    given a mapping returns a `mappingproxy` instance that is **read-only** but a **dynamic-view** 
    of the original mapping. 

    This means that the original mapping can be seen through `mappingproxy`
    but changes cannot be made through it.

    ```python

    ```
