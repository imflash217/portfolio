<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Tensorflow Tutorial

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

### Initialize with zeros & ones
We will use `tf.ones()` and `tf.zeros()` to initialize a tensor of shape `shape`,
where all elements are either zeros or ones

```python

def ones(shape, dtype=tf.int64):
    """Creates a tensor of ones with shape=shape
    Args:
        shape: the shape of the resulting tensor
        dtype: the datatype of every element in the resulting tensor
    Returns:
        A tensor where all elements are 1
    """
    ## Create ones tensor using `tf.ones()`
    ones = tf.ones(shape, dtype=dtype)

    ## Create a session
    sess = tf.Session()
    
    ## Execute the op in the session to calculate its value
    ones = sess.run(ones)

    ## Close the session
    sess.close()

    ## Return the ones tensor
    return ones
```
```python
ones_tensor = ones([2,3])
print(ones_tensor)
```
```
[[1,1,1],
 [1,1,1]]
```

## Building a Neural Network

### Building the model

```python
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow.python.framework import ops

def model(X_train, Y_train, X_test, Y_test, 
        lr=1e-3, num_epochs=1500, bs=32, verbose=True):
        """
        Implements a 3-layer Tensorflow Neural Network:
        [Linear]->[Relu]->[Linear]->[Relu]->[Linear]-[Softmax]
          
        Args:
            X_train: the train dataset inputs
            Y_train: the train dataset labels
            X_test: the test dataset inputs
            Y_test: the test dataset labels
            lr: the learnign rate
            num_epochs: number of epochs
            bs: batch-size
            verbose: True if you want to print the process else False
        
        Returns:
            the trained model parameters.
        """
        
        ops.reset_default_graph()       ## to be able to rerun the model, w/o overwriting the tf.variables
        tf.set_random_seed(217)         ## to keep consistent results
        seed = 3                        ## to keep consistent results
        (n_x, m) = X_train.shape        ## n_x = input size; m = number of training examples
        n_y = Y_train.shape[0]          ## n_y = output size
        costs = []                      ## to keep track of the costs

        ## Step-1: Create placeholders of shape = (n_x, n_y)
        X, Y = create_placeholders(n_x, n_y)
        
        ## Step-2: Initialize parameters
        parameters = initialize_parameters()

        ## Step-3: Forward propagation
        ##         Build the forward propagation the tf graph
        Z3 = forward_proagation(X, parameters)

        ## Step-4: Cost function
        ##         Add cost function to tf graph
        cost = compute_cost(Z3, Y)

        ## Step-5: Backward propagation
        ##         Define the tf optimizer. Use `AdamOptimizer`
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
        
        ## Step-6: Initialize all variables
        init = tf.global_variables_initializer()

        ## Step-7: Start the session to compute the tf graph
        with tf.Session() as sess:
            ## Step-7.1: Run the initializer `init`
            sess.run(init)

            ## Step-7.2: Do the training loop
            for epoch in range(num_epchs):
                epoch_cost = 0.0        ## Define the cost for each epoch
                num_batches = m // bs
                seed += 1
                minibatches = random_mini_batches(X_train, Y_train, bs, seed)
                for (Xb, Yb) in minibatches:
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:Xb, Y:Yb})
                    epoch_cost += minibatch_cost
                epoch_cost /= num_batches

            ## Step-8: Save the trained model parameters
            parameters = sess.run(parameters)
            print("parameters have been trained")

            ## Step-9: How to calculate the correct predictions & accuracy
            correct_preds = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))

            ## Step-10: Calculate the train & test accuracies
            accuracy_train = accuracy.eval({X:X_train, Y:Y_train})
            accuracy_test = accuracy.eval({X:X_test, Y:Y_test})

            return parameters
```

