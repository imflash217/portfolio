<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Hash Tables
`Author: Vinay Kumar (@imflash217) | Date: 29/January/2021`

<!-- ######################################################################################################### -->

## `Definition`

???+ note "Definition"
    **Hash Table** is a data structure which stores data in an **associative manner** (i.e. in a (key, value) pair).

    - In a hash table, the data is stored in an array format where each data-value has its own unique index-value. Due to this feature, the access to data becomes very fast if we know the desired index-value; irrespective of the size of the data.
    - Hash Table uses an array as a storage medium and uses **hashing** to generate the index where an element is to be inserted or to be located from.

<!-- ######################################################################################################### -->

## `Hashing`

???+ note "Hashing"
    ![Hashing](https://www.tutorialspoint.com/data_structures_algorithms/images/hash_function.jpg)

    Hashing is a technique to **map a range of keys into a range of indexes** (usually of an array).

    - A very generic hashing function is **modulo operator** (`x % y`).

<!-- ######################################################################################################### -->

### `Example`

???+ quote "Example of Hashing"
    - Consider a hash-table of `size=20`
    - Following (`key`, `value`) pairs to be stored using the hash-table

    ```python
    dict = {9: 20,
            12: 70,
            42: 80,
            7: 25,
            2: 21}
    ```

    | Key | Hash           | Array index |
    | --- | -------------- | ----------- |
    | 9   | `9 % 20 = 9`   | `9`         |
    | 12  | `12 % 20 = 12` | `12`        |
    | 42  | `42 % 20 = 2`  | `2`         |
    | 7   | `7 % 20 = 7`   | `7`         |
    | 2   | `2 % 20 = 2`   | `2`         |

    As we can see that a given **hashing function** can create the same hash-value from two different keys. (in above table keys `42` and `2`). So we use **`Linear Probing`** to resolve conflicts.

<!-- ######################################################################################################### -->

## `Linear Probing`

???+ note "Linear Probing"
    **Linear Probing** is a method used to resolve conflicts in the hash-value. It may happen that the hash-function creates an already used index of the array. In such case we search the next empty location of the array **by looking into the next cell until we find an empty cell**

    So in our above example, the updated hash-table would map `key = 2` to `index = 3`:

    | Key | Hash           | Array index |
    | --- | -------------- | ----------- |
    | 9   | `9 % 20 = 9`   | `9`         |
    | 12  | `12 % 20 = 12` | `12`        |
    | 42  | `42 % 20 = 2`  | `2`         |
    | 7   | `7 % 20 = 7`   | `7`         |
    | 2   | `2 % 20 = 2`   | **`3`**     |

## `Search`

???+ success "search() method for hash-table"
    **`Search`**

<!-- ######################################################################################################### -->

## `Python Implementation`

```python
--8<-- "../ProgrammingContests/ctci/hashtable.py"
```

<!-- ######################################################################################################### -->

???+ quote "Author Disclaimer"
    `Author: Vinay Kumar (@imflash217) | Date: 29/January/2021`

    The contents of this article were originally published at the references below. I have assembled it for my own understanding. Feel free to reuse and tag along the references. :+1:

## `References`
[^1]: https://www.hackerearth.com/practice/data-structures/hash-tables/basics-of-hash-tables/tutorial/
[^2]: https://www.tutorialspoint.com/python_data_structure/python_hash_table.htm
[^3]: https://www.tutorialspoint.com/data_structures_algorithms/hash_data_structure.htm
[^4]: http://blog.chapagain.com.np/hash-table-implementation-in-python-data-structures-algorithms/
[^5]: 

<!-- ######################################################################################################### -->