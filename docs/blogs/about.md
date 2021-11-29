---
hide:
  - toc        # Hide table of contents
---

<!-- - navigation # Hide navigation -->

## Tips & Tricks

???+ done "LR Scheduler"
    - [x] Similar to the `learning rate`, the `lr-scheduler` to apply depends on the 
            classifier & the model.
    - [x] For image classifiers and **`SGD` optimizer**, the **`Multi-Step LR Scheduler`**
            is shown to be a good choice.
    - [x] Models trained with **`Adam`** commonly use a smooth exponential-decay in the `lr` or a cosine-like scheduler.
    - [x] For TRANSFORMERS:
        - :rotating_light: Remember to use a **`learning rate WARMUP`**
        - :rotating_light: The `cosine-scheduler` is often used for decaying the `lr` afterwards (but can also be replaced by `exponential decay`)



------------------------------------------------------------------------------
## References
- [ ] Hyperparameter Search
    - [ ] Learning rate finder: https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html
    - [ ] Auto Scaling batch sizes: https://pytorch-lightning.readthedocs.io/en/latest/training_tricks.html#auto-scaling-of-batch-size
    - [ ] Compare hyperparam search performance in TensorBoard: https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
    - [ ] $3^{rd}$ party libraries: https://medium.com/pytorch/accelerate-your-hyperparameter-optimization-with-pytorchs-ecosystem-tools-bc17001b9a49
    - [ ] Saving `git` hash metadata: https://github.com/Nithin-Holla/meme_challenge/blob/f4dc2079acb78ae30caaa31e112c4c210f93bf27/utils/save.py#L26

