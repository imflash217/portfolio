<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Deep Learning Interviews

<!-- ######################################################################################################### -->

## Introduction

---------------
### Q1: Distribution of maximum entropy

> What is the distribution of **maximum entropy**; i.e. the distribution that has the maximum entropy among all distributions in a bounded interval `[a, b]`, `(-\inf, +\inf)`?

Solution:

In a bounded interval `[a, b]`, the **UNIFORM DISTRIBUTION** has the maximum entropy. The variance of the Uniform Distribution $\mathcal{U}(a, b)$ is $\sigma^2 = \frac{(b-a)^2}{12}$.
Therefore, the maximum entropy in a bounded interval `[a, b]` is $\frac{\log{12}}{2} + \log(\sigma)$

----------

### Q2: What's the purpose of this code-snippet?

> Describe in your own words. what is the purpose of this code snippet?
> ``` python
> self.transforms = []
> if rotate:
>     self.transforms.append(RandomRotate())
> if flip:
>     self.transforms.append(RandomFLip())
> ```

Solution:

**Overfitting** is a common problem that occurs during training of machine learning systems. Among various strategies to overcome the problem of overfitting; **data-augmentation** is a very handy method. **Data Augmentation** is a regularization technique that synthetically expands the data-set by utilizing *label-preserving* transformations to add more **invariant examples** of the same data samples. Data Augmentation is very important in balancing the data distribution across various classes in the datatset. Some of the data-augmentation techniques are: `random rotation`, `cropping`, `random flip`, `zooming`etc.

Usually, the data-augmentation process is done in the CPU before uploading the batched-data for training the model on the GPU.