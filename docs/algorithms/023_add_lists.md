<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

`Author: Vinay Kumar (@imflash217) | Update: 07/October/2021`

<!-- ######################################################################################################### -->

## `23: Add lists`

???+ danger "Add lists"
    Write in a function that takes head of two **linked lists**, 
    each representing a number. The nodes of the linked-lists contain digits as value.
    The nodes in the input lists are **reversed** 
    (i.e. the least significant digit of the number is head).

    The function should return the head of the new linked list 
    representing the sum of the input lists.
    The output should have its digits reversed as well.

    ```
    Say we wanted to compute 621 + 354 normally. The sum is 975:

       621
     + 354
     -----
       975

    Then, the reversed linked list format of this problem would appear as:

        1 -> 2 -> 6
     +  4 -> 5 -> 3
     --------------
        5 -> 7 -> 9
    ```
