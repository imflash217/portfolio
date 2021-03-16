<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Arrays
`Author: Vinay Kumar (@imflash217) | Date: 15/March/2021`

<!-- ######################################################################################################### -->

## `Definition`

???+ note "Definition"
    An **Array** is an ordered list of data that we access with a numerical index.
    Generally speaking an array is allocated upfront as a single block of memory based
    on the number of elements and type of the data we want the array to hold. This allows
    us to **read** and **write** elements into the array efficiently, since our program
    knows exactly where each element is stored in the memory.

    On the other hand **removing**, **adding** and **finding arbitrary values** in an array
    can be a linear-time operation.

    **Removing** or **splicing** requires shifting all elements by one to fill the gap.

    **Inserting** a new element would requires shifting or allocating a new larger array to hold the elements.

    **Finding** an element in the array would require iterating over the entire array in the worst-case.

## `Dynamic Array`

???+ note "Dynamic Arrays"
    It is worth noting that in many **statically-typed** programming languages (e.g. Java, C++);
    an array is limited to its *initially declared size*.

    **ALL modern languages** support DYNAMICALLY-SIZED arrays; which automatically increase or decrease their size
    by allocating a new copy of the array when it begins to run out-of-memory.

    Dynamic arrays guarantee better **amortized performance** by only performing these costly operations when necessary.

## `Calculating Memory Usage`

???+ note "Calculating Memory Usage"
    To calculate the memory usage of an array simply multiply the size of the array with the size of the data-type.

    ???+ question "What is the memory usage of an array that contains one-thousand 32-bit integers?"
        ```
        1000 * 32 bits  = 1000 * 4 bytes = 4000 bytes = 4Kb
        ```
    ???+ question "What is the memory usage of an array that contains one-hundred 10-char strings?"
        ```
        100 * 10 chars = 100 * 10 * 1 byte = 1000 bytes = 1Kb
        ```

???+ success "Common Array Operations"

    - **Insert** an item
    - **Remove** an item
    - **Update** an item
    - **Find** an item
    - **Loop** over array
    - **Copy** an array
    - **Copy-part-of-the-array**
    - **Sort** an array
    - **Reverse** an array
    - **Swap** an array
    - **Filter** an array

???+ danger "When to use an array in an interview"
    Use an array when you need dta in an ordered list with **fast-indexing** or **compact-memory-footprint**.

    Don't use an array if you need to search for unsorted items efficiently or insert and remove items frequently.

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

## `Delete`

???+ danger "delete() method for hash-table"
    **`Delete`**
<!-- ######################################################################################################### -->

## `Python Implementation`

```python
--8<-- "../ProgrammingContests/ctci/hashtable.py"
```

<!-- ######################################################################################################### -->

???+ quote "Author Disclaimer"
    `Author: Vinay Kumar (@imflash217)`

    `Date: 29/January/2021`

    The contents of this article were originally published at the references below. I have assembled it for my own understanding. Feel free to reuse and tag along the references. :+1:

## `References`
[^1]: https://www.hackerearth.com/practice/data-structures/hash-tables/basics-of-hash-tables/tutorial/
[^2]: https://www.tutorialspoint.com/python_data_structure/python_hash_table.htm
[^3]: https://www.tutorialspoint.com/data_structures_algorithms/hash_data_structure.htm
[^4]: http://blog.chapagain.com.np/hash-table-implementation-in-python-data-structures-algorithms/
[^5]: https://runestone.academy/runestone/books/published/pythonds/SortSearch/Hashing.html
[^6]: http://paulmouzas.github.io/2014/12/31/implementing-a-hash-table.html

<!-- ######################################################################################################### -->
