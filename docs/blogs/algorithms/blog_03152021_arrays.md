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

<!-- ######################################################################################################### -->
