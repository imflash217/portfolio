## Linked List

???+ quote "Question #1"
    > What two properties are typically stored in the nodes of a singly linked list?

    :trophy: **`value`** and **`next`**

???+ quote "Question #2"
    > What terms are commonly used to describe the **first node** & **last node** of a linked list?

    :trophy: **HEAD** for first-node 
    :trophy: **TAIL** for last-node

???+ quote "Question #4"
    > What is the **dummy head pattern** for a linked-list?
    
    :trophy: The **dummy head pattern** is where we use a fake node to 
    act as the HEAD of the linked-list. The dummy-head is used to simplify
    **edge cases** such as inserting teh first node into an empty linked-list.

???+ quote "Question #4"
    > Why might the expression **`current_node.next.val`** be UNSAFE?

    :trophy: If the current node is a **TAIL-node** then its `.next` will be `None`
    and `None` object does not have `.val` attribute.
