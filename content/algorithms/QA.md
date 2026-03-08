---
title: "Linked List Q&A"
tags:
  - algorithms
  - linked-list
  - interview
summary: "Interview-style Q&A covering linked list fundamentals  -  nodes, pointers, head/tail, and common operations."
---

# Linked List

**Question #1**
> What two properties are typically stored in the nodes of a singly linked list?

🏆 **`value`** and **`next`**

---

**Question #2**
> What terms are commonly used to describe the **first node** & **last node** of a linked list?

🏆 **HEAD** for first-node

🏆 **TAIL** for last-node

---

**Question #3**
> What is the **dummy head pattern** for a linked-list?

🏆 The **dummy head pattern** is where we use a fake node to
act as the HEAD of the linked-list. The dummy-head is used to simplify
**edge cases** such as inserting the first node into an empty linked-list.

---

**Question #4**
> Why might the expression **`current_node.next.val`** be UNSAFE?

🏆 If the current node is a **TAIL-node** then its `.next` will be `None`
and `None` object does not have `.val` attribute.

---

**Question #5**
> What are the time complexities for the following linked-list operations?
> - Lookup
> - Insert
> - Delete

🏆
| Operation | Time Complexity |
|-----------|----------------|
| Lookup    | $O(n)$         |
| Insert    | $O(1)$         |
| Delete    | $O(1)$         |