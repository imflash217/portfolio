---
title: "Tensorflow Tutorial"
tags:
  - tensorflow
  - tutorial
  - deep-learning
summary: "Hands-on TensorFlow v1.0 tutorial  -  variables, sessions, training algorithms, and building a neural network."
---

# Tensorflow Tutorial

In this session you will learn to do the following in `TensorFlow v1.0`

1. Initialize Variables
2. Start your own session
3. Train Algorithms
4. Implement a Neural Network

## Exploring the Tensorflow Library

### Example-1: General Overview

```python
import tensorflow as tf

y_hat = tf.constant(36, name="y_hat")           ## Defines a "y_hat" constant. Sets its value to 36
y = tf.constant(39, name="y")                   ## Defines a "y" constant. Sets its value to 39
loss = tf.Variable((y-y_hat)**2, name="loss")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(loss)
    print(loss)
```

Writing and running programs in `Tensorflow` has the following steps:

1. **Create tensors** (variables) that are not yet evaluated/executed.
2. **Write operations** between those tensors.
3. **Initialize** the tensors.
4. **Create a Session**.
5. **Run the session**. This will run the operations written in step-2.

### Example-2: `tf.Session()`

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
the **computation graph**; but we haven't run this computation yet!

```python
sess = tf.Session()
print(sess.run(c))
```
```
20
```

### Example-3: `tf.placeholder()`

A **placeholder** is an object whose value we can specify ONLY later.

```python
sess = tf.Session()
x = tf.placeholder(tf.int64, name="x")
print(sess.run(2*x, feed_dict={x:9}))
sess.close()
```
```
18
```

### Using one-hot encodings:

```python
def one_hot_matrix(labels, num_classes):
    num_classes = tf.constant(num_classes, name="num_classes")
    one_hot_matrix = tf.one_hot(indices=labels, depth=num_classes, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
```

## Building a Neural Network

### Building the model

```python
def model(X_train, Y_train, X_test, Y_test,
        lr=1e-3, num_epochs=1500, bs=32, verbose=True):
        """
        Implements a 3-layer Tensorflow Neural Network:
        [Linear]->[Relu]->[Linear]->[Relu]->[Linear]->[Softmax]
        """
        ops.reset_default_graph()
        tf.set_random_seed(217)
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []

        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                epoch_cost = 0.0
                num_batches = m // bs
                minibatches = random_mini_batches(X_train, Y_train, bs)
                for (Xb, Yb) in minibatches:
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:Xb, Y:Yb})
                    epoch_cost += minibatch_cost
                epoch_cost /= num_batches

            parameters = sess.run(parameters)
            correct_preds = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))
            return parameters
```
