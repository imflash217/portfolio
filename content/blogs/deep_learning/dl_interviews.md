---
title: "🧑🏻‍💻 DL Interviews"
tags:
  - deep-learning
  - interviews
summary: "DL interview prep  -  maximum entropy, data augmentation, overfitting, and probability of success problems."
---

# Deep Learning Interviews

## Introduction

### Q1: Distribution of maximum entropy

> What is the distribution of **maximum entropy**; i.e. the distribution that has the maximum entropy among all distributions in a bounded interval `[a, b]`, `(-\inf, +\inf)`?

**Solution:**
In a bounded interval `[a, b]`, the **UNIFORM DISTRIBUTION** has the maximum entropy. The variance of the Uniform Distribution $\mathcal{U}(a, b)$ is $\sigma^2 = \frac{(b-a)^2}{12}$.
Therefore, the maximum entropy in a bounded interval `[a, b]` is $\left(\frac{\log{12}}{2} + \log(\sigma)\right)$

---

### Q2: What's the purpose of this code-snippet?

> Describe in your own words, what is the purpose of this code snippet?
> ```python
> self.transforms = []
> if rotate:
>     self.transforms.append(RandomRotate())
> if flip:
>     self.transforms.append(RandomFlip())
> ```

**Solution:**
**Overfitting** is a common problem that occurs during training of machine learning systems. Among various strategies to overcome the problem of overfitting; **data-augmentation** is a very handy method. **Data Augmentation** is a regularization technique that synthetically expands the data-set by utilizing *label-preserving* transformations to add more **invariant examples** of the same data samples.

Usually, the data-augmentation process is done in the CPU before uploading the batched-data for training the model on the GPU.

---

## Logistic Regression

### Q3: Drawbacks of model fitting

> For a fixed number of observations in a dataset, introducing more number of variables normally generate a model that has a better fit to the data. What may be drawbacks of such a model fitting strategy?

**Solution:**
Introducing more number of variables increases the capacity of the model. If the number of data points in the dataset is kept fixed, and then increasing the number of model parameters (variables) leads to **OVERFITTING**. Overfitting is a scenario where the trained model performs very well on the training data but performs poorly on the test dataset due to lack of generalization capabilities as the overly sized model just remembered the data points instead of understanding the features & data distribution in the training set.

---

### Q4: Odds of Success

> Define the term **`odds of success`**, both *qualitatively* and *formally*.

**Solution:**
**Odds of Success** of an event in an experiment is the ratio of *probability of the event occurring* and the *probability of the event not occurring*
i.e. $\left(\frac{\text{probability of occurrence of an event E}}{1 - \text{(probability of the occurrence of the event E)}}\right)$
