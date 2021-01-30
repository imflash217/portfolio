# Lecture `#1`

- [ ] AI v/s ML v/s DL (Venn Diagram)
- [ ] One-hot-encoding
- [x] Loss Function
- [ ] Training v/s Evaluation Error
- [ ] Model Selection
- [ ] Hyperparams
- [ ] Overfitting v/s Underfitting
- [ ] Generalization Gap
- [ ] Model Capacity
- [ ] K-fold Cross Validation
  - [ ] Leave-one-out Cross Validation
- [ ] 

???+ note "What is Machine Learning?"
    It is a field that aims to extract relationships and structures in the data.
    `Example: How to map data to annotations?`

???+ note "Loss Function"
    We need a **measure** to see how well our system is doing at learning.
    This measure is called **Loss Function**

    - [x] Sum-of-Squared-Error (SSE): $2^2$ $\sum_{i}\normalize{(y_i - f(x_i)}_2^2$

???+ note "Training"
    The process of teaching our system to minimize errors is called as **Training**.

???+ note "Evaluation"
    The process of determining the _performance_ of our trained system over an **unseen dataset** is called as **Evaluation**.

???+ note "Unsupervised Learning"
    - [x] Generative Models (GAN, AE, RBM)
    - [x] Latent Variable Modeling (PCA, AE)
    - [ ] Clustering

???+ success "[special case of] Cross Validation "
    If there are many point on the graph of CV($\theta$) with similar values near the minimum; we choose the most **parsimonious** model that has a CV value within the _standard deviation_ from the best model $\theta^*$.

    In other words; we pick the first $\theta$ for which the CV value satisfies
    $$CV(\theta) < CV(\theta^*) + std(CV(\theta^*))$$