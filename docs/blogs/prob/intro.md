# Probablity Theory

A key concept in the field of **pattern recognition** is that of **uncertainity**.
It arrises through:

1. noise on measurements
2. finite size of datasets.

Probability theory provides a consistent framework for the quantification 
and manipulation of uncertainity and forms one of the central foundations
of Pattern Recognition.

When combined with **Decision Theory**, it allows us to make optimal predictions
given all the informtion available to us, even though that information may be incomplete or ambiguous.

## Example: Red/Blue boxes

### The setup

Imagine we have two boxes `red_box` and `blue_box`. 
Each box contains different number of fruits `oranges` and `apples`.
```
red_box:
    - apples: 2
    - oranges: 6
blue_box:
    - apples: 3
    - oranges: 1
```

### The Experiment

Now, we do the following steps:

1. _Select a box_: **Randomly** pick one of the boxes (either `red_box` or `blue_box`)
2. _Pick a fruit_: Then, **randomly** select an item of fruit for the box.
3. _Replace the fruit_: Having observed the type of the picked fruit (`apple` or `orange`),
now, replace it in the box from which it came.
4. _Repeat_ steps 1-to-3 multiple times.

### The Pre-conditions / Assumptions

Now, let's suppose in doing the above experiment, 

1. we pick the `red_box` **40%** of the times and the `blue_box` **60%** of the times. 
2. Also, assume that that when we remove an item of fruit
from a box, we are equally likely to select any of the pices of fruits in the box.


