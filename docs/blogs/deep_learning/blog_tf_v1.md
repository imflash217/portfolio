<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Tensorflow Tutorial
`Author: Vinay Kumar (@imflash217) | Date: 03/October/2021`

<!-- ######################################################################################################### -->

In this session you will learn to do the following in `TensorFlow v1.0`

1. Initialize Variables
2. Start your own session
3. Train Algorithms
4. Implement a Neural Network

<!-- ######################################################################################################### -->

## Exploring the Tensorflow Library

### Example-1: General Overview

```python
import tensorflow as tf

y_hat = tf.constant(36, name="y_hat")           ## Defins a "y_hat" constant. Sets its value to 36
y = tf.constant(39, name="y")                   ## Defins a "y" constant. Sets its value to 39
loss = tf.Variable((y-y_hat)**2, name="loss")

init = tf.global_variables_initializer()        ## Used to initialize the variables with the
                                                ## respective values when "sess.run(init)" is called

with tf.Session() as sess:                      ## Creates a session to execute our program
    sess.run(init)                              ## initializes the global variables
    sess.run(loss)                              ## executes the program stored in "loss" variable
    print(loss)                                 ## prints the value stored in "loss" variable
```

Writing and running programs in `Tensorflow` has the following steps:

1. **Create tensors** (variables) that are not yet evaluated/executed.
2. **Write operations** between those tensors.
3. **Initialize** the tensors.
4. **Create a Session**.
5. **Run the session**. This will run the operations written in step-2.

So, when we created a variable for `loss`, we simply defined the loss as a function of other
quantities but did not evaluate its value. To evaluate it, we had to run 
`tf.global_variables_initializer()` to intialize the values and then inside `sess.run(init)`
we calculated the updated value and prited it in the last line above.

### Example-2: `tf.Session()`

Now, let's take a look at

```python
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
```

```
Tensor("Mul:0", shape=(), dtype=int32)
```

As expected we will not see `20`. We got a tensor saying that the result of the tensor
does not have the `shape` attribute and is of the type `int32`. All we did was to put in
the **computation graph**; but we haven't run this computation yet! In order to actually 
multiply the two numbers we have to create a sessiona nd run it.

```python
sess = tf.Session()
print(sess.run(c))
```
```
20
```

**Awesome!!**. To summarize, remember the following:

1. Initialize your variables.
2. Create a session.
3. Run the operations inside the session.


### Example-3: `tf.placeholder()`

Next, we will see how to use a placeholder.

A **placeholder** is an object whose value we can specify ONLY later.

To specify values for a placeholder, we can pass in values by using a 
"feed dictionary" (`feed_dict` variable).

```python
## Below we create a placeholder for x.
## This allows us to pass in a number later when we run the SESSION

sess = tf.Session()

x = tf.placeholder(tf.int64, name="x")      ## the placeholder variable
print(sess.run(2*x, feed_dict={x:9}))

sess.close()
```
```
18
```

### Using one-hot encodings:

```python
def one_hot_matrix(labels, num_classes):
    """
    Creates a matrix where the i-th row corresponds to the ith class number.
    j-th column corresponds to the j-th example.
    So, if the label for j-th example is i; then only the ith value is 1 in j-th column
    
    Args:
        labels: the labels for each example
        num_classes: the number of classes in this task
    Returns:
        a one-hot matrix
    """
    ## create a tf.constant & name it "num_classes"
    num_classes = tf.constant(num_classes, name="num_classes")
    
    ## Use tf.one_hot (be careful with "axis")
    one_hot_matrix = tf.one_hot(indices=labels, depth=num_classes, axis=0)

    ## Create a session
    sess = tf.Session()

    ## Execute the one_hot_matrix graph inside the session
    one_hot = sess.run(one_hot_matrix)

    ## Close the session
    sess.close()
    
    ## return the one_hot matrix
    return one_hot

```
```python
import numpy as np

labels = np.array([1,2,0,1,2,2,3])
num_classes = 4
one_hot = one_hot_matrix(labels, num_classes)
print(one_hot)
```
```
[[0,0,1,0,0,0,0],
 [1,0,0,1,0,0,0],
 [0,1,0,0,1,1,0],
 [0,0,0,0,0,0,1]]
```
