---
title: "The Transformer Family Version 2.0"
date: 2023-01-27
author: Lilian Weng
tags:
  - architecture
  - attention
  - transformer
  - foundation
  - long-read
  - reinforcement-learning
summary: "A comprehensive overview of the Transformer architecture and its many variants, covering attention mechanisms, positional encoding, efficient attention patterns, and applications to reinforcement learning."
---

Many new Transformer architecture improvements have been proposed since the original "The Transformer Family" post about three years ago. Here we do a big refactoring and enrichment  -  restructuring the hierarchy of sections and improving many sections with more recent papers. Version 2.0 is a superset of the old version, about twice the length.

## Notations

| Symbol | Meaning |
|---|---|
| $d$ | The model size / hidden state dimension / positional encoding size |
| $h$ | The number of heads in multi-head attention layer |
| $L$ | The segment length of input sequence |
| $N$ | The total number of attention layers in the model |
| $\mathbf{X} \in \mathbb{R}^{L \times d}$ | The input sequence where each element has been mapped into an embedding vector of shape $d$ |
| $\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$ | The key weight matrix |
| $\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$ | The query weight matrix |
| $\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$ | The value weight matrix |

## Transformer Basics

The **Transformer** (Vaswani, et al., 2017) model has an encoder-decoder architecture, as commonly used in many NMT models. Later simplified Transformer was shown to achieve great performance in language modeling tasks, like in encoder-only BERT or decoder-only GPT.

### Attention and Self-Attention

**Attention** is a mechanism in neural network that a model can learn to make predictions by selectively attending to a given set of data. The amount of attention is quantified by learned weights and thus the output is usually formed as a weighted average.

**Self-attention** is a type of attention mechanism where the model makes prediction for one part of a data sample using other parts of the observation about the same sample.

Transformer relies on the *scaled dot-product attention*: given a query matrix $\mathbf{Q}$, a key matrix $\mathbf{K}$ and a value matrix $\mathbf{V}$, the output is a weighted sum of the value vectors:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

For a query and a key vector $\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$, we have a scalar score:

$$a_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d_k}}\right) = \frac{\exp\left(\frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d_k}}\right)}{\sum_{r \in S_i} \exp\left(\frac{\mathbf{q}_i \mathbf{k}_r^\top}{\sqrt{d_k}}\right)}$$

### Multi-Head Self-Attention

The **multi-head self-attention** module is a key component in Transformer. Rather than only computing the attention once, the multi-head mechanism splits the inputs into smaller chunks and then computes the scaled dot-product attention over each subspace in parallel:

$$\text{MultiHead}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) = [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o$$

$$\text{where head}_i = \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)$$

### Positional Encoding

Because self-attention operation is permutation invariant, it is important to use proper **positional encoding** to provide order information to the model.

#### Sinusoidal Positional Encoding

Sinusoidal positional encoding is defined as follows, given the token position $i$ and the dimension $\delta$:

$$\text{PE}(i, \delta) = \begin{cases} \sin\left(\frac{i}{10000^{2\delta'/d}}\right) & \text{if } \delta = 2\delta' \\ \cos\left(\frac{i}{10000^{2\delta'/d}}\right) & \text{if } \delta = 2\delta' + 1 \end{cases}$$

#### Rotary Position Embedding

Rotary position embedding (RoPE; Su et al. 2021) encodes the absolute position with a rotation matrix and multiplies key and value matrices of every attention layer with it to inject relative positional information at every layer.

Given a vector $\mathbf{z}$, to rotate it counterclockwise by $\theta$, we multiply by a rotation matrix:

$$R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

When generalizing to higher dimensional space, RoPE divides the $d$-dimensional space into $d/2$ subspaces and constructs a block-diagonal rotation matrix. Then both key and query matrices incorporate positional information:

$$\mathbf{q}_i^\top \mathbf{k}_j = (\mathbf{R}^d_{\Theta, i} \mathbf{W}^q \mathbf{x}_i)^\top (\mathbf{R}^d_{\Theta, j} \mathbf{W}^k \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{W}^q \mathbf{R}^d_{\Theta, j-i} \mathbf{W}^k \mathbf{x}_j$$

## Efficient Attention

The computation and memory cost of vanilla Transformer grows quadratically with sequence length and hence it is hard to be applied on very long sequences.

### Sparse Attention Patterns

A simple alteration to make self-attention less expensive is to restrict the attention span of each token to **local** context only, so that self-attention grows linearly with the sequence length.

**Sparse Transformer** (Child et al., 2019) introduced *factorized self-attention*, through sparse matrix factorization, making it possible to train dense attention networks with hundreds of layers on sequence length up to 16,384.

### Low-Rank Attention

**Linformer** (Wang et al. 2020) approximates the full attention matrix with a *low rank* matrix, reducing the time & space complexity to be linear:

$$\overline{\text{head}}_i = \text{softmax}\left(\frac{\mathbf{X}_q \mathbf{W}^q_i (\mathbf{E}_i \mathbf{X}_k \mathbf{W}^k_i)^\top}{\sqrt{d}}\right) \mathbf{F}_i \mathbf{X}_v \mathbf{W}^v_i$$

### Content-based Attention

The **Reformer** (Kitaev, et al. 2020) replaces the dot-product attention with *locality-sensitive hashing (LSH) attention*, reducing the complexity from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \log L)$.

A hashing scheme $x \mapsto h(x)$ is *locality-sensitive* if it preserves the distancing information between data points. The Reformer adopts: $h(x) = \arg\max([xR; -xR])$ where $\mathbf{R} \in \mathbb{R}^{d \times b/2}$ is a fixed random matrix.

## Longer Context

The length of an input sequence for transformer models at inference time is upper-bounded by the context length used for training. Naively increasing context length leads to high consumption in both time $\mathcal{O}(L^2 d)$ and memory $\mathcal{O}(L^2)$.

### Context Memory

**Transformer-XL** (Dai et al., 2019) modifies the architecture to reuse hidden states between segments with an additional memory:

$$\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)} = [\text{stop-gradient}(\mathbf{h}_{\tau}^{(n-1)}) \circ \mathbf{h}_{\tau+1}^{(n-1)}]$$

$$\mathbf{Q}_{\tau+1}^{(n)} = \mathbf{h}_{\tau+1}^{(n-1)} \mathbf{W}^q$$

$$\mathbf{K}_{\tau+1}^{(n)} = \widetilde{\mathbf{h}}_{\tau+1}^{(n-1)} \mathbf{W}^k$$

## Transformers for Reinforcement Learning

The self-attention mechanism avoids compressing the whole past into a fixed-size hidden state and does not suffer from vanishing or exploding gradients as much as RNNs.

**Decision Transformer** (DT; Chen et al 2021) formulates RL problems as a process of *conditional sequence modeling*, outputting the optimal actions conditioned on the desired return:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T)$$

Three linear layers are added for return-to-go, state, and action respectively. The prediction head learns to predict $a_t$ corresponding to the input token $s_t$.

## Citation

Cited as:

> Weng, Lilian. (Jan 2023). The Transformer Family Version 2.0. Lil'Log. https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/.

## References

[1] Vaswani et al. "Attention is all you need." NIPS 2017.

[2] Su et al. "Roformer: Enhanced transformer with rotary position embedding." arXiv preprint arXiv:2104.09864 (2021).

[3] Child et al. "Generating Long Sequences with Sparse Transformers" arXiv:1904.10509 (2019).

[4] Wang et al. "Linformer: Self-Attention with Linear Complexity." arXiv preprint arXiv:2006.04768 (2020).

[5] Kitaev et al. "Reformer: The Efficient Transformer" ICLR 2020.

[6] Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." ACL 2019.

[7] Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling" arXiv preprint arXiv:2106.01345 (2021).

[8] Press et al. "Train Short, Test Long: Attention With Linear Biases Enables Input Length Extrapolation." ICLR 2022.
