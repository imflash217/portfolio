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
