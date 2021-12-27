# Naïve Bayes Classifiers

In this article we will talk about **Multinomial Naïve Bayes Classifier**,
so called because it is a *Bayesian Classifier that makes a simplifying (naïve)
assumption about the interaaction b/w features*.

Let's understand the intuition of this calssifier in the context of **text classification**.
Given a text document we first respresnt the text document as a **bag-of-words** 
(i.e. an unordered set of words in the document with their position information removed)
keeping only the **word-frequency** in the given document.
In this **bag-of-words** representation, all we care about is how many times a given word appears in this document.

> **Naïve Bayes** is a _probabilistic classifier_, meaning that for a given document **`d`**,
out of all classes $c\in C$ the classifier returns the class $\hat{c}$ which has the maximum
posterior probability given the document **`d`**.
