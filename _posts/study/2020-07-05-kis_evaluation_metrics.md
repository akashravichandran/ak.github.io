---
title: Understanding Evaluation Metrics, one last time!
date: 2020-07-05
tags: [python, accuracy, evaluation metric, study]
description: Explaining the evaluation metrics used in ml
categories: keepitsimple
---

In order to understand how our model is performing, we need something called as accuracy to evaluate it. But can accuracy alone answer all our questions about how good a model is. Absolutely not, that is why we need other
metrics like sensitivity, specificity, predictive values, and the ROC curve, which helps in understanding our model in a better way.

---

**Accuracy - total examples that the model correctly classified** 

Let's work with an example on why accuracy is not the only metric to look upon when evaluating our model. To explain it, I am taking a medical scenario where a given patient is classified as either diseased (POSITIVE) or normal (NEGATIVE).

![Imgur](https://i.imgur.com/6FmFuBd.png)

Cosider the above table to be a test set of 10 patients. Eight of them have the ground truth of **Normal** and two have the ground truth of **Disease**. Let's say we have a model which outputs **Normal** for all 10 patients. This is of course not a useful model, but notice that it's getting all the normal patients right. So it's getting eight out of 10 patients right, and thus has an accuracy of 0.8.

![Imgur](https://i.imgur.com/NQByqrj.png)

Now let us consider model 2 which correctly predicts **Disease** on the two disease patients here, and also calls two of the **Normal** patients as **Disease**. Now we can compute the accuracy of Model 2 and if we go through this computation, we will find once again that we get eight out of 10 patients right with model 2 to get an accuracy of 0.8. 

**So we have two models with the same accuracy of 0.8 but with different predictive power. Although we haven't formalized this, we get a sense that model 2 is perhaps doing something more useful than model 1 because it is at least attempting to distinguish between Normal and disease.**

---

Let us derive other useful evaluation metrics such as sensitivity and specificity to define our model better.

---

In terms of Probability, 

**Accuracy = probability of being correct**

But, what does correctness mean:

**Accuracy** = probability that the model predicts positive and a patient has the disease $$P(+\: \cap \:disease)$$ + probability that the model predicts negative and the patient is normal $$P(-\: \cap \:normal)$$

---

In terms of conditional probability, we can write this as:

$$P(A \cap B) = P(A \mid B)\:P(B)$$

probability of A and B is the probability of A given B times the probability of B. 

---

Thus, we can interpret accuracy as, 

**Accuracy** =  $$P(+\: \cap \:disease)\: P(disease) \:+\: P(-\: \cap \:normal)\: P(normal)$$

**Accuracy** = $$Sensitivity * Prevalence + Specificity * (1 - Prevalance)$$

**Sensitivity** = True Postive Rate = $$P(+\: \cap \:disease)\: P(disease)$$ = the probability that the model classifies a patient as having the disease given that they have the disease

**Specificity** = True Negative Rate =  $$P(-\: \cap \:normal)\: P(normal)$$ = the probability that the model classifies a patient as being normal given that they are normal

Thus, we've broken our global measure of accuracy down into the useful quantities of sensitivity and specificity.

**Prevalence** = $$P(disease)$$ = the probability of a patient having disease in a population

**1 - Prevalence** = $$P(normal)$$ = the probability of being normal is simply one minus the probability of having disease 

---

> We can thus write accuracy in terms of sensitivity, specificity, and prevalence. **Why is this useful?** This equation allows us to see accuracy as a weighted average of sensitivity and specificity. 

> The weight associated with the sensitivity is the prevalence, and the weight associated with the specificity is one minus the prevalence. It will help us find out any of these quantities given the other three quantities are provided.

---

Let's try to understand this using an example: 

![Imgur](https://i.imgur.com/3zjRdzL.png)

Sensitivity can be computed as the fraction of disease examples that are also positive. So in the denominator, we have the number of disease examples, which is 2 and in the numerator the number of positive and disease examples which is only one. So we have 1 out of 2 or 0.50.

Specificity can be computed as the fraction of normal examples that are also negative. So in the denominator, we have the number of normal examples which is going to be 8 and in the numerator, the number of negative and normal examples, this is going to be 6 and this is 6 over 8 to get 0.75. 

So that's the computation of sensitivity and specificity. 

Now let's look at the prevalence of disease in this set. This is simply the fraction of disease examples computed as the number of disease examples over the total number of examples which is 2 over 10 or 0.2. 

> Now, using the relationship between sensitivity, specificity, and prevalence, we can also get to the accuracy. 

So this is going to be equal sensitivity 0.5 times the prevalence 0.2, plus the specificity 0.75 times 1 minus prevalence which is 0.8. And thus, we get an accuracy of 0.7 same as the green coloured in the table above.

---

Sensitivity tells us given we know a patient has a disease, what is the probability that the model predicts positive? 

> But we are more interested in knowing if the model predicts positive on a patient, what is the probability that they actually have the disease? This is called the positive predictive value or PPV of the model.

Similarly, while specificity asks, what is the probability the model predicts negative, given a patient is normal? 

> We are interested in knowing the probability that a patient is normal, given the model prediction is negative. This is called the negative predictive value or NPV of a model. 

Let's compute the **Positive Predictive Value - PPV** and **Negative Predictive Value - NPV** on an example. Once again, we have ten examples on which a model has made its predictions. First, let's compute PPV. 

![Imgur](https://i.imgur.com/Gh2WihC.png)

PPV can be computed as the fraction of positive examples that are also disease. So in the denominator, we'll have the number of positive which is three. And in the numerator, the number that is one positive and disease. So 1 over 3, or 0.33. 

NPV can be computed as the fraction of negative examples that are also normals. So in the denominator, we're looking at everywhere the model prediction is negative, it's seven. And in the numerator all the places where it's negative and the ground truth is normal which is six, so this 6 over 7, or 0.85. 

---

Now that we've seen PPV and NPV, in addition to sensitivity and specificity, let's look at how they relate to each other.

Rewriting PPV

$$PPV = P(pos | \hat{pos})$$

($$pos$$ is "actually positive" and $$\hat{pos}$$ is "predicted positive").

By Bayes rule, this is

$$PPV = \frac{P(\hat{pos} | pos) \times P(pos)}{P(\hat{pos})}$$ 


For the numerator:

$$Sensitivity = P(\hat{pos} | pos)$$

> Sensitivity is how well the model predicts actual positive cases as positive.

$$Prevalence = P(pos)$$ 

> Prevalence is how many actual positives there are in the population.

For the denominator:

$$P(\hat{pos}) = TruePos + FalsePos$$ 

> In other words, the model's positive predictions are the sum of when it correctly predicts positive and incorrectly predicts positive.

The true positives can be written in terms of sensitivity and prevalence.

$$TruePos = P(\hat{pos} | pos) \times P(pos)$$

and you can use substitution to get

$$TruePos = Sensitivity \times Prevalence$$

The false positives can also be written in terms of specificity and prevalence:

$$FalsePos = P(\hat{pos} | neg) \times P(neg)$$

$$1 - specificity = P(\hat{pos} | neg )$$

$$1 - prevalence = P(neg)$$

PPV rewritten:

If you substitute these into the PPV equation, you'll get

$$PPV = \frac{sensitivity \times prevalence}{sensitivity \times prevalence + (1 - specificity) \times (1 - prevalence)}$$

---

Thus we have understood various ways to interpret our model's predictive capability. In the future update, will include confusion matrix and discuss on how to interpret our model from that. 

---

I hope that this blog post was insightful. I look forward to your feedback in the comments section. Happy reading!

---


​	

