<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Deep Learning Interviews

<!-- ######################################################################################################### -->

## Introduction

### Q1: Distribution of maximum entropy

> What is the distribution of **maximum entropy**; i.e. the distribution that has the maximum entropy among all distributions in a bounded interval `[a, b]`, `(-\inf, +\inf)`?

In a bounded interval `[a, b]`, the **UNIFORM DISTRIBUTION** has the maximum entropy. The variance of the Uniform Distribution $\mathcal{U}(a, b)$ is $\square{(\sigma)} = \square{(b-a)}/12$. 
Therefore, the maximum entropy in a bounded interval `[a, b]` is $$\frac{\log(12)}{2} + \log(\sigma)$$