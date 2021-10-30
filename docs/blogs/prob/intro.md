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

### The Theory

In this experiemnt:

1. the identity of the box to be chosen is a **random variable** $B$ which can take 
two possible values `r` and `b` (corresponding to red or blue boxes).
2. Similarly, the identity of fruit is also a **random variable** $F$ and it can
take either of the values `a` and `o` (corresponding to apple an dorange respectively).

> Definition:
>
> The **Probability of an event** is defined as the _fraction_ of times, that event occurs
> out of the total number of trials (in the limit that the total number of trials goes to
> infinity). All probabilities must lie in the interval [0, 1]

Thus , the probability of selecting the `red_box` is $4/10$ and that of the `blue_box` is
$6/10$.

$$
p(B=r) = 4/10
$$

$$
p(B=b) = 6/10
$$

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

