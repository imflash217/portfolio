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

??? success "Solution"
    In a bounded interval `[a, b]`, the **UNIFORM DISTRIBUTION** has the maximum entropy. The variance of the Uniform Distribution $\mathcal{U}(a, b)$ is $\sigma^2 = \frac{(b-a)^2}{12}$.
    Therefore, the maximum entropy in a bounded interval `[a, b]` is $\left(\frac{\log{12}}{2} + \log(\sigma)\right)$

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

??? success "Solution"
    **Overfitting** is a common problem that occurs during training of machine learning systems. Among various strategies to overcome the problem of overfitting; **data-augmentation** is a very handy method. **Data Augmentation** is a regularization technique that synthetically expands the data-set by utilizing *label-preserving* transformations to add more **invariant examples** of the same data samples. Data Augmentation is very important in balancing the data distribution across various classes in the datatset. Some of the data-augmentation techniques are: `random rotation`, `cropping`, `random flip`, `zooming`etc.

    Usually, the data-augmentation process is done in the CPU before uploading the batched-data for training the model on the GPU.

------------

## Logistic Regression

### Q3: Drawbacks of model fitting

> For a fixed number of observations in a dataset, introducing more number of variables normally generate a model that has a better fit to the data. What may be drawbacks of such a model fitting strategy?

??? success "Solution"
    Introducing more number of variables increasing the capacity of the model. If the number of data points in teh dataset is kept fixed, and then increasing the number of model parameters (variables) leads to **OVERFITTING**. Overfitting is a scenario where the trained model performs very well on the training data but performs poorly in the test daatset due to lack of generalization capabilites as the overly sized model just remembered the data points instead of understanding teh features & data distribution in the traning set.

??? quote "हिन्दी में"
    > **प्रश्न:** 
    > Training data में data-points की संख्या निश्चित रखते हुए, model-variables की संख्या बढ़ाने से हम एक ऐसा trained model को प्राप्त कर सकते हैं जो training data को अत्यंत आची प्रकार fit कर सकता है। इस प्रक्रिया के करने से क्या-क्या हानियाँ होते हैं?

    **उत्तर:**
    Model के variables बढ़ाने से model की capacity बढ़ती है और इससे वो training data को आचे से समझ सकता है और training data-distribution के बारे में समझ सकता है। परंतु अगर variables की संख्या बढ़ाने के साथ-साथ अगर हम training-data की संख्या नहीं बढ़ाते हैं तो trained model में एक विकृति होने लगती है जिसे हम **OVERFITTING** कहते हैं। Overfitting होने से हमारा trained model, training-data के विषय में तो बहुत अच्छे से जानता है परंतु वो test-data पर वो अच्छे से कार्य नहीं करता है क्योंकि training के समय वो अपनी generalization क्षमता को विक्षित नहीं कर पाया और संभवतः एक रटंत (memorized training data) model ही बन पाया। अतः हमे *overfitting* को यथा संभव रोकने का प्रयास करना चाहिए जिसके लिये अनेक पथ अपनाए जाते हैं जैसे: (१) dropout (२) data augmentation (३) pruning

---------------

### Q4: Odds of Success

> Define the term **`odds of success`**, both *qualitively* and *formally*.
> Give a numerical example that stresses the relationship the relationship between *probability* and *odds of an event occuring*.

??? success "Solution"
    **Odds of Success** of an event in an experiment is the ratio of *probability of the event occuring* and the *probability of the event not occuring*
    i.e. $\left(\frac{\text{probability of occurance of an event E}}{1 - \text{(probability of the occurance of the event E)}}\right)$

