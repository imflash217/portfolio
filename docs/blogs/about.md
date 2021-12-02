<!--
---
hide:
    - toc        # Hide table of contents
    - navigation # Hide navigation 
---
-->

# The Preface of the key technological stuffs here

## Tips & Tricks

### LR Scheduler
- [x] Similar to the `learning rate`, the `lr-scheduler` to apply depends on the 
    classifier & the model.
- [x] For image classifiers and **`SGD` optimizer**, the **`Multi-Step LR Scheduler`**
    is shown to be a good choice.
- [x] Models trained with **`Adam`** commonly use a smooth exponential-decay 
    in the `lr` or a cosine-like scheduler.
- [x] For TRANSFORMERS:
    - :rotating_light: Remember to use a **`learning rate WARMUP`**
    - :rotating_light: The `cosine-scheduler` is often used for decaying the `lr` 
        afterwards (but can also be replaced by `exponential decay`)

### Regularizaation
- [x] Regularization is important in networks when we see a significantly higher 
    **training** performance than **test** performance.
- [x] The regularization parameters all interact with each other and hence 
    **must be tuned together**. The most commonly used regularization techniques are:
    - **`Weight Decaay`**
    - **`Dropout`**
    - **`Augmentation`**
- [x] Dropout is a good regularization technique as it has shown to be
    applicable on most architectures and has shown to **reduce overfitting**.
- [x] If you want to use **weight-decay in Adam**, use **`torch.optim.AdamW`** instead of `torch.optim.Adam`.
- [x] Domain specific regularization: There are a couple of regularization techniques that 
    depend on the input-data / domain as shown below.
    - :rotating_light: Computer Vision: Image augmenatation like 
        - **`horizontal_flip`**, 
        - **`rotation`**, 
        - **`scale_and_crop`**, 
        - **`color_distortion`**, 
        - **`gaussian_noise`** etc.
    - :rotating_light: NLP: input dropout of **whole words**
    - :rotating_light: Graphs: 
        - Dropping edges
        - Dropping nodes
        - Dropping part of the features of all nodes

## Debugging in PyTorch

### Under-performing model

???+ danger "Situation/Problem"
    Your model is not reaching the performance it should, 
    but PyTorch is not telling you why that happens!! These are very annoying bugs.

------------------------------------------------------------------------------

#### Softmax, CrossEntropy & NLLLoss

:trophy: The most common mistake is the mismatch between the loss function
and the output activations. A very usual common source of confusion is the relationship 
between **`nn.Softmax`, `nn.LogSoftmax`, `nn.NLLLoss`, & `nn.CrossEntropyLoss`**

1. **`nn.CrossEntropyLoss`** does two operations on its inputs: **`nn.LogSoftmax`** & **`nn.NLLLoss`**.
Hence, the input to the `nn.CrossEntropyLoss` should be the output of the last layer of the network.

    :rotating_light: **Don't apply `nn.Softmax` before the `nn.CrossEntropyLoss`.** 
    Otherwise, PyTorch will apply the Softmax TWICE which will signifacntly worsen the performance.

2. If you use **`nn.NLLLoss`**, you need to apply **log-softmax** before yourselves.
`nn.NLLLoss` requires **log-probabilities** as its input not just plain *probabilities*. 
So, make sure to use `F.log_softmax()` instead of `nn.Softmax`

------------------------------------------------------------------------------

#### Softmax over correct dimension/axis

Be careful to apply softmax over correct dimensio/axis in your output.
For eg. you apply softamx over **last dimension** like this: **`nn.Softmax(dim=-1)`**

------------------------------------------------------------------------------

#### Categorical Data & Embeddings


#### Hidden size mismatch

If you perform matrix multiplications and have a shape mismatch between two matrices,
PyTorch will contain and throw error. 

However, there are situations where PyTorch does not throw any error because the misaligned
dimensions have (unluckily) the same dimension. For example, imagine you have a weight matrix
**`W`** of shape **`[d_in, d_out]`**. If you take an inout **`x`** of shape **`[batch_size, d_in]`**.
And you want to do the matrix multiplication as **`out = W.matmul(x)`** then the shape of the output `out` 
will be correct as **`[batch_size, d_out]`**. But, suppose if by chance **`batch_size == d_in`**
then both **`W.matmul(x)`** and **`x.matmul(W)`** will produce the same sized output `[d_in, d_out]`.
This is definitely not the behaviour we want as it hides the error in the order of 
matrix maultiplication over different dimension.

:rotating_light: So, **always test your code with multiple different batch sizes to prevent
shape misalignments with the batch dimension**.


### Use nn.Sequential & nn.ModuleList

If you have a model with lots of layers, you might waant to summarize them into 
`nn.Sequential` or `nn.ModuleList` object. In the forward pass, you only need to call the 
`Sequential` or iterate through the `ModuleList`.

A multi-layer-perceptron (MLP) can be implemented as follows:

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_dims=64,
        hidden_dims=[128, 256],
        output_dims=10,
    ):
        super().__init__()
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i+1],
                nn.ReLU(inplace=True)
            ]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
```

### In-place Activation functions

Some activation functions such as **`nn.ReLU`** and **`nn.LeakyReLU`** have an argument **`inplace`**.
By default, it is set to **`False`**, but it is highly recommended to set it to **`True`** in neural networks.

> Setting it to `True`, makes the original value of the **input** overridden by the **new output** during the
> forward pass.

:trophy: This option of `inplace` is ONLY available to activations functions
**where we don't need to know the original input for backpropagation.**

For example, in **`nn.ReLU`**, the value sthat are set to zero have a gradient of ZERO independent
of the specific input values.

:rotating_light: In-place operations can save a lot of memory, especially if you have a very large feature map.






------------------------------------------------------------------------------


## References
- [ ] Hyperparameter Search
    - [ ] Learning rate finder: 
        https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html#learning-rate-finder
    - [ ] Auto Scaling batch sizes: 
        https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#auto-scaling-of-batch-size
    - [ ] Compare hyperparam search performance in TensorBoard: 
        https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
    - [ ] $3^{rd}$ party libraries: 
        https://medium.com/pytorch/accelerate-your-hyperparameter-optimization-with-pytorchs-ecosystem-tools-bc17001b9a49
    - [ ] Saving `git` hash metadata: 
        https://github.com/Nithin-Holla/meme_challenge/blob/f4dc2079acb78ae30caaa31e112c4c210f93bf27/utils/save.py#L26

- [x] PyTorch Tutorials: https://effectivemachinelearning.com/

