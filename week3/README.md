##  Classification and Representation
  
  
###  Classification
  
  
Examples of Classifications problems:
  
* Email: Spam / Not Spam
* Online Transactions: Fraudulent (Yes/No)
* Tumor: Malignant / Benign
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?y%20&#x5C;in%20&#x5C;lbrace%200,1%20&#x5C;rbrace%20&#x5C;tag{%200:%20Negative%20class,%201:%20Positive%20class}"/></p>  
  
  
**How do we develop a classification algorithm?**
  
I think that we could do is to apply the algorithm that we already know:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20&#x5C;theta^T%20&#x5C;ast%20x"/></p>  
  
  
![](images/classificationExplanation.jpg )
  
If you want to make predictions, one thing that we could do is to use a threshold classifier output at 0.5:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20&#x5C;begin{cases}%20%20%201%20&amp;&#x5C;text{if%20}%20%20h_&#x5C;theta{(x)}%20&#x5C;ge%200.5%20&#x5C;&#x5C;%20%20%200%20&amp;&#x5C;text{if%20}%20%20h_&#x5C;theta{(x)}%20&lt;%200.5&#x5C;end{cases}"/></p>  
  
  
  
**Note:** For classification we know that y is either zero or one. But if you are using linear regression where the hypothesis can output values that are much larger than one or less than zero, even if all of your training examples have labels y equals zero or one.
  
###  Hypothesos Representation
  
  
We would like our classifier to output values between zero and one.
  
When we were using linear regression our hypothesis had the form of:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20&#x5C;theta^T%20&#x5C;ast%20x"/></p>  
  
  
For logistic regression we will modify this a little bit and make the hypothesis g of theta transpose x. 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20g(&#x5C;theta^T%20&#x5C;ast%20x)&#x5C;text{,%20where%20}%20g(z)=%20&#x5C;cfrac{a}{1%20+%20{e^{-z}}}"/></p>  
  
  
**Sigmoid \ Logistic Function**
  
The sigmoid function, while it asymptotes at one and asymptotes at zero, as a z axis, the horizontal axis is z. As z goes to minus infinity, g(z) approaches zero. And as g(z) approaches infinity, g(z) approaches one. And so because g(z) upwards values are between zero and one, we also have that h(x) must be between zero and one.
  
![Hypothesis](images/LRhypothesis.jpg )
  
**Intepretation of Hypothesis Output**
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20&#x5C;text{estimated%20probability%20that%20y=1%20on%20input%20x}"/></p>  
  
  
Example:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;text{if%20}%20x%20=%20&#x5C;begin{bmatrix}%20x_0%20&#x5C;&#x5C;%20x_1%20&#x5C;end{bmatrix}%20=%20&#x5C;begin{bmatrix}%201%20&#x5C;&#x5C;%20tumorSize%20&#x5C;end{bmatrix}%20&#x5C;newline%20~%20&#x5C;newlineh_&#x5C;theta{(x)}%20=%200.7"/></p>  
  
  
h_theta will give us the probability that our output is 1. For example, h_theta = 0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{(x)}%20=%20p(y=1|x;&#x5C;theta)%20&#x5C;text{%20probability%20that%20y=1,%20given%20x,%20parameterized%20by%20}%20&#x5C;theta%20&#x5C;newline&#x5C;begin{aligned}&amp;%20h_&#x5C;theta(x)%20=%20P(y=1%20|%20x%20;%20&#x5C;theta)%20=%201%20-%20P(y=0%20|%20x%20;%20&#x5C;theta)%20&#x5C;&#x5C;%20&amp;%20P(y%20=%200%20|%20x;&#x5C;theta)%20+%20P(y%20=%201%20|%20x%20;%20&#x5C;theta)%20=%201&#x5C;end{aligned}"/></p>  
  
  
##  Decision Boundary
  
  
In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}&amp;%20h_&#x5C;theta(x)%20&#x5C;geq%200.5%20&#x5C;rightarrow%20y%20=%201%20&#x5C;&#x5C;&amp;%20h_&#x5C;theta(x)%20&lt;%200.5%20&#x5C;rightarrow%20y%20=%200%20&#x5C;&#x5C;&#x5C;end{aligned}"/></p>  
  
  
The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}&amp;%20g(z)%20&#x5C;geq%200.5%20&#x5C;&#x5C;%20&amp;%20when%20&#x5C;;%20z%20&#x5C;geq%200&#x5C;end{aligned}"/></p>  
  
  
Remember.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20z=0,%20e^{0}=1%20&#x5C;Rightarrow%20g(z)=1&#x2F;2%20&#x5C;&#x5C;%20z%20&#x5C;to%20&#x5C;infty,%20e^{-&#x5C;infty}%20&#x5C;to%200%20&#x5C;Rightarrow%20g(z)=1%20&#x5C;&#x5C;%20z%20&#x5C;to%20-&#x5C;infty,%20e^{&#x5C;infty}&#x5C;to%20&#x5C;infty%20&#x5C;Rightarrow%20g(z)=0%20&#x5C;end{aligned}"/></p>  
  
  
So if our input to g is theta^T X, then that means:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20&amp;%20h_&#x5C;theta(x)%20=%20g(&#x5C;theta^T%20x)%20&#x5C;geq%200.5%20&#x5C;&#x5C;%20&amp;%20when%20&#x5C;;%20&#x5C;theta^T%20x%20&#x5C;geq%200&#x5C;end{aligned}"/></p>  
  
  
From these statements we can now say:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20&amp;%20&#x5C;theta^T%20x%20&#x5C;geq%200%20&#x5C;Rightarrow%20y%20=%201%20&#x5C;&#x5C;%20&amp;%20&#x5C;theta^T%20x%20&lt;%200%20&#x5C;Rightarrow%20y%20=%200%20&#x5C;&#x5C;%20&#x5C;end{aligned}"/></p>  
  
  
The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.
  
**Example:**
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20&amp;%20&#x5C;theta%20=%20&#x5C;begin{bmatrix}5%20&#x5C;&#x5C;%20-1%20&#x5C;&#x5C;%200&#x5C;end{bmatrix}%20&#x5C;&#x5C;%20&amp;%20y%20=%201%20&#x5C;;%20if%20&#x5C;;%205%20+%20(-1)%20x_1%20+%200%20x_2%20&#x5C;geq%200%20&#x5C;&#x5C;%20&amp;%205%20-%20x_1%20&#x5C;geq%200%20&#x5C;&#x5C;%20&amp;%20-%20x_1%20&#x5C;geq%20-5%20&#x5C;&#x5C;%20&amp;%20x_1%20&#x5C;leq%205%20&#x5C;&#x5C;%20&#x5C;end{aligned}"/></p>  
  
  
In this case, our decision boundary is a straight vertical line placed on the graph where x_1 = 5, and everything to the left of that denotes y = 1, while everything to the right denotes y = 0.
  
Again, the input to the sigmoid function g(z) (e.g. theta^T X) doesn't need to be linear, and could be a function that describes a circle (e.g. z = theta_0 + theta_1 x_1^2 + theta_2 x_2^2 ) or any shape to fit our data.
  
![](images/NOlinearDecisionBoundaries.jpg )
  