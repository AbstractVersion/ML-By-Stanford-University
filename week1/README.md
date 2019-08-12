## Introduction

 **What is machine learning**

Even among machine learning practitiones, there isn't a well accepted definition of what is and what isn't machine learning.

**Definitions**

_Arthur Samuel (1959)_

> Machine Learning is the field of study that gives computers the ability to learn without being explicitly learned.

_Tom Mitchell(1998) - Better one_

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Example: Playing checkers

* E: the experience of playing many games of checkers
* T: the task of playing checkers
* P: the probability that the program will win the next game

---

**Question**

Suppose your email program wtacges which emails you do or do not mark as spam, and based on that learns how to better filter spam. What is task T in this setting?

_Answer:_ Classifying emails as spam or not spam.

---

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning and Unsupervised learning.

### Supervised learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

Example 1:

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2:

(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

---

**Question**

You're running a company, and you want to develop learning algorithms to address each of two problems.

Problem 1: You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months.

Problem 2: You'd like software to examine indiviudal customer accouncts, and for each account decide if it has been hacked/compromised.

Should you treat these as a classification or regression problem?

_Answer:_ Treat prblem 1 as a regression problem and problem 2 as a classification problem.

---

## Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

Example:

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

---

**Question**

Of the following examples, which would you address using an unsupervised learning algorithm? (Check all that apply.)

_Answer:_ 

Correct: 

* Given a set of news articles found on the web, group them into sets of articles about the same stories.
* Given a database of customer data, automatically discover market segments and group customers into different market segments.

Un-Correct:

* Given email labeled as spam/not spam, learn a spam filter.
* Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.


---

### Model and Cost Function

**Model Representation**

To establish notation for future use, we’ll use x<sup>(i)</sup> to denote the “input” variables (living area in this example), also called input features, and y<sup>(i)</sup> to denote the “output” or target variable that we are trying to predict (price).

A pair (x<sup>(i)</sup>, y<sup>(i)</sup>) is called a training example, and the dataset that we’ll be using to learn—a list of m training examples (x<sup>(i)</sup>, y<sup>(i)</sup>); i=1,..,m is called a training set. Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

![Hypothesis Schema](./images/trainingSchema.png)

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.