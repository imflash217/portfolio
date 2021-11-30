<!--
---
hide:
    - toc        # Hide table of contents
    - navigation # Hide navigation 
---
-->

## Tips & Tricks

???+ done "LR Scheduler"
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

???+ danger "Regularizaation"
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

