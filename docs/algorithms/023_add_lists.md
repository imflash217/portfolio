<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

<!-- ######################################################################################################### -->

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

???+ done "Solution"
    ```python
    class Node:
        def __init__(self, val):
            super().__init__()
            self.val = val
            self.next = None

    def add_lists(head_1, head_2, carry=0):
        """Recursive solution"""
        ## base case:
        if head_1 is None and head_2 is None and carry == 0:
            return None

        ## grab the values of each node 
        ## or use a dummy value 0 if the node is None
        
        val_1 = head_1.val if head_1 else 0
        val_2 = head_2.val if head_2 else 0

        _sum = val_1 + val_2 + carry                    ## add the two values
        digit = _sum % 10                               ## accounting for carry (next line)
        carry = 1 if _sum >= 10 else 0

        result = Node(digit)                            ## create a new "Node" with new digit
        
        next_1 = head_1.next if head_1 else None
        next_2 = head_2.next if head_2 else None

        result.next = add_lists(next_1, next_2, carry)  ## recursive call
        return result

    ```
