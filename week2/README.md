##  Multivariate Linear Regression
  
  
###  Multiple Features
  
  
_**Note: [7:25 - theta<sup>T</sup> is a 1 by (n+1) matrix and not an (n+1) by 1 matrix]**_
  
Linear regression with multiple variables is also known as "multivariate linear regression".
  
We now introduce notation for equations where we can have any number of input variables.
  
  
| Size (feet^2)      | Number of bedrooms | Number of floors | Age of home (years) | Price ($1000) |
| ------------------ | ------------------ | ---------------- | ------------------- | ------------- |
| 2104               | 5                  | 1                | 45                  | 460           |
| 1416               | 3                  | 2                | 40                  | 232           |
| 1534               | 3                  | 2                | 30                  | 315           |
| 852                | 2                  | 1                | 36                  | 178           |
| ...                | ...                | ...              | ...                 | ...           |
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20x_j^{(i)}%20&amp;=%20&#x5C;text{value%20of%20feature%20}%20j%20&#x5C;text{%20in%20the%20}i^{th}&#x5C;text{%20training%20example}%20&#x5C;&#x5C;%20x^{(i)}&amp;%20=%20&#x5C;text{the%20input%20(features)%20of%20the%20}i^{th}&#x5C;text{%20training%20example}%20&#x5C;&#x5C;%20m%20&amp;=%20&#x5C;text{the%20number%20of%20training%20examples}%20&#x5C;&#x5C;%20n%20&amp;=%20&#x5C;text{the%20number%20of%20features}%20&#x5C;end{aligned}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?x^{(2)}%20=%20&#x5C;begin{bmatrix}%201416%20&#x5C;&#x5C;%203%20&#x5C;&#x5C;%202%20&#x5C;&#x5C;%2040%20&#x5C;end{bmatrix}"/></p>  
  
  
That's an index into my training set. This is not X to the power of 2. Instead, this is, you know, an index that says look at the second row of this table. This refers to my second training example.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?x^{(2)}_3%20=%202"/></p>  
  
  
With this notation X2 is a four dimensional vector. In fact, more generally, this is an in-dimensional feature back there. With this notation, X2 is now a vector and so, I'm going to use also Xi subscript J to denote the value of the J, of feature number J and the training example.
  
---
  
**Question**
  
In the training set above, what is <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;nobreak%20x_1^{(4)}&#x5C;text{?}"/></p>  
  
  
**Answer**
  
* The size (in feet^2) of the 1st home in the training set
* The age (in years) of the 1st home in the training set
* *(**Correct**) The size (in feet^2) of the 4th home in the training set*
* The age (in years) of the 4th home in the training set
  
---
  
**Form of hypothesis**
  
Previous form: 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;xcancel{h_{&#x5C;theta}{x}%20=&#x5C;theta_0%20+%20&#x5C;theta_1%20&#x5C;ast%20x}"/></p>  
  
  
Previously this was the form of our hypothesis, where x was our single feature, but now that we have multiple features, we aren't going to use the simple representation any more. Instead, a form of the hypothesis in linear regression is going to be this, can be theta 0 plus theta 1 x1 plus theta 2 x2 plus theta 3x3 plus theta 4 X4. And if we have N features then rather than summing up over our four features, we would have a sum over our N features.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?h_{&#x5C;theta}{x}%20=&#x5C;theta_0%20+%20&#x5C;theta_1%20&#x5C;ast%20x_1%20%20+%20&#x5C;theta_2%20&#x5C;ast%20x_2%20+%20&#x5C;dotsb%20+%20%20+%20&#x5C;theta_n%20&#x5C;ast%20x_n"/></p>  
  
  
For convieniece of notation, define x<sub>0</sub> = 1. So now my feature vector X becomes this N+1 dimensional:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?x%20=%20&#x5C;begin{bmatrix}%20x_0%20&#x5C;&#x5C;%20x_1%20&#x5C;&#x5C;%20%20x_2%20&#x5C;&#x5C;%20&#x5C;dotsb%20&#x5C;&#x5C;%20x_n%20&#x5C;end{bmatrix}&#x5C;in&#x5C;R^{n+1}"/></p>  
  
  
My parameter vector would be:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta%20=%20&#x5C;begin{bmatrix}%20&#x5C;theta_0%20&#x5C;&#x5C;%20&#x5C;theta_1%20&#x5C;&#x5C;%20&#x5C;theta_2%20&#x5C;&#x5C;%20&#x5C;dotsb%20&#x5C;&#x5C;%20&#x5C;theta_n%20&#x5C;end{bmatrix}&#x5C;in&#x5C;R^{n+1}"/></p>  
  
  
So we conclude with the hypothesis:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}h_&#x5C;theta(x)%20=&#x5C;begin{bmatrix}&#x5C;theta_0%20&#x5C;hspace{2em}%20&#x5C;theta_1%20&#x5C;hspace{2em}%20...%20&#x5C;hspace{2em}%20&#x5C;theta_n&#x5C;end{bmatrix}&#x5C;begin{bmatrix}x_0%20&#x5C;&#x5C;%20x_1%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20x_n&#x5C;end{bmatrix}=%20&#x5C;theta^T%20x&#x5C;end{aligned}"/></p>  
  
  
###  Gradient Descent for Multiple 
  
  
The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}%20&amp;%20&#x5C;text{repeat%20until%20convergence:}%20&#x5C;;%20&#x5C;lbrace%20&#x5C;&#x5C;%20&#x5C;;%20&amp;%20&#x5C;theta_0%20:=%20&#x5C;theta_0%20-%20&#x5C;alpha%20&#x5C;frac{1}{m}%20&#x5C;sum&#x5C;limits_{i=1}^{m}%20(h_&#x5C;theta(x^{(i)})%20-%20y^{(i)})%20&#x5C;cdot%20x_0^{(i)}&#x5C;&#x5C;%20&#x5C;;%20&amp;%20&#x5C;theta_1%20:=%20&#x5C;theta_1%20-%20&#x5C;alpha%20&#x5C;frac{1}{m}%20&#x5C;sum&#x5C;limits_{i=1}^{m}%20(h_&#x5C;theta(x^{(i)})%20-%20y^{(i)})%20&#x5C;cdot%20x_1^{(i)}%20&#x5C;&#x5C;%20&#x5C;;%20&amp;%20&#x5C;theta_2%20:=%20&#x5C;theta_2%20-%20&#x5C;alpha%20&#x5C;frac{1}{m}%20&#x5C;sum&#x5C;limits_{i=1}^{m}%20(h_&#x5C;theta(x^{(i)})%20-%20y^{(i)})%20&#x5C;cdot%20x_2^{(i)}%20&#x5C;&#x5C;%20&amp;%20&#x5C;cdots%20&#x5C;&#x5C;%20&#x5C;rbrace%20&#x5C;end{aligned}"/></p>  
  
  
  
In other words:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}&amp;%20&#x5C;text{repeat%20until%20convergence:}%20&#x5C;;%20&#x5C;lbrace%20&#x5C;&#x5C;%20&#x5C;;%20&amp;%20&#x5C;theta_j%20:=%20&#x5C;theta_j%20-%20&#x5C;alpha%20&#x5C;frac{1}{m}%20&#x5C;sum&#x5C;limits_{i=1}^{m}%20(h_&#x5C;theta(x^{(i)})%20-%20y^{(i)})%20&#x5C;cdot%20x_j^{(i)}%20&#x5C;;%20&amp;%20&#x5C;text{for%20j%20:=%200...n}&#x5C;&#x5C;%20&#x5C;rbrace&#x5C;end{aligned}"/></p>  
  
  
---
  
**Question**
  
When there are n features, we define the cost function as:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta_0)%20=%20&#x5C;dfrac%20{1}{2m}%20&#x5C;sum%20_{i=1}^m%20(h_&#x5C;theta%20(x^{(i)})%20-%20y^{(i)})^2"/></p>  
  
  
For linear regression, which of the following are also equivalent and correct definitions of J(θ)?
  
**Answer**
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta)%20=%20&#x5C;frac{1}{2m}&#x5C;sum_{i%20=%201}^{m}(%20&#x5C;theta^{T}x^{(i)}%20-%20y^{(i)})^{2}"/></p>  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(%20&#x5C;theta)%20=%20&#x5C;frac{1}{2m}&#x5C;sum_{i%20=%201}^{m}((%20&#x5C;sum_{j%20=%200}^{n}&#x5C;theta_{j}x_{j}^{(i)})%20-%20y^{(i)})^{2}"/></p>  
  
  
---
  
The following image compares gradient descent with one variable to gradient descent with multiple variables:
  
![Gradient descent with one variable to gradient descent with multiple variables](images/GDmVariable.png )
  
###  Gradient Descent in Practice I - Feature Scaling
  
  
**Idea:** If you have a problem where you have multiple features, if you make sure that the features are on a similar scale, by which I mean make sure that the different features take on similar ranges of values, then gradient descents can converge more quickly.
  
For Example:
  
![Feature Scaling](images/GDInPractice01.png )
  
If you plot the contours of the cos function J of theta, then the contours may look like this, where, let's see, J of theta is a function of parameters theta zero, theta one and theta two. I'm going to ignore theta zero, so let's about theta 0 and pretend as a function of only theta 1 and theta 2, but if x1 can take on them, you know, much larger range of values and x2 It turns out that the contours of the cause function J of theta can take on this very very skewed elliptical shape.
  
And if you run gradient descents on this cos-function, your gradients may end up taking a long time and can oscillate back and forth and take a long time before it can finally find its way to the global minimum.
  
Concretely if you instead define the feature X one to be the size of the house divided by two thousand, and define X two to be maybe the number of bedrooms divided by five, then the count well as of the cost function J can become much more, much less skewed so the contours may look more like circles.
  
**Selection of the range**
  
Get every feautre int aproimately a  -1 <= x<sub>i</sub> <= 1 range.Ideally:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?−1%20&#x5C;le%20x_{(i)}%20&#x5C;le%201"/></p>  
  
  
or
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?−0.5%20&#x5C;le%20x_{(i)}%20&#x5C;le%200.5"/></p>  
  
  
**Mean normalization**
  
Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?x_i%20:=%20&#x5C;frac{x_i%20-%20&#x5C;mu_i}{s_i}"/></p>  
  
  
Where mu is the **average**  of all the values for feature (i) and s_i is the range of values (max - min), or s_i is the standard deviation. Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.
  
For example, if x_i  represents housing prices with a range of 100 to 2000 and a mean value of 1000, then,
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?x_i%20:=%20&#x5C;frac{price%20-%201000}{1900}"/></p>  
  
  
  
---
  
**Question**
  
Suppose you are using a learning algorithm to estimate the price of houses in a city. You want one of your features x_i to capture the age of the house. In your training set, all of your houses have an age between 30 and 50 years, with an average age of 38 years. Which of the following would you use as features, assuming you use feature scaling and mean normalization?
  
  
**Answer:** <p align="center"><img src="https://latex.codecogs.com/gif.latex?x_i%20:=%20&#x5C;frac{&#x5C;text{age%20of%20house}%20-%2038}{20}"/></p>  
  
  
  
---
  
###  Gradient Descent in Practice II - Learning Rate
  
  
Concretely, here's the gradient descent update rule. And what we want to do is to develop a debugging rule, and some tips for making sure that gradient descent is working correctly. And second, how to choose the learning rate alpha or at least how I go about choosing it. 
  
Here's something that I often do to make sure that gradient descent is working correctly. The job of gradient descent is to find the value of theta for you that hopefully minimizes the cost function J(theta). What I often do is therefore plot the cost function J(theta) as gradient descent runs. So the x axis here is a number of iterations of gradient descent and as gradient descent runs you hopefully get a plot that maybe looks like this.
  
![Gradient Descent in Practice II](images/GradientDescentinPracticeII.jpg )
  
***Warning: The J(theta) should decrease after every iteration.***
  
It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.
  
  
![Gradient Descent in Practice II](images/GradientDescentinPracticeII02.jpg )
  
To summarize:
  
* If alpha is too small: slow convergence.
* If alpha is too large: ￼may not decrease on every iteration and thus may not converge.
  
---
  
**Question**
  
Suppose a friend ran gradient descent three times, with alpha = 0.01, alpha = 0.1, and alpha = 1, and got the following three plots (labeled A, B, and C):
  
![Gradient Descent in Practice II](images/GDPracticeQuizplots.jpg )
  
Which plots corresponds to which values of alpha?
  
*Answer:* A is alpha=0.1, B is alpha = 0.001, C is alpha =1. 
  
> In graph C, the cost function is increasing, so the learning rate is set too high. Both graphs A and B converge to an optimum of the cost function, but graph B does so very slowly, so its learning rate is set too low. Graph A lies between the two.
  
---
  
To choose alpha, try an array of range of values: ..., 0.001, 0.003, 0.01, 0.03, 0,1, 0.3, 1, ...
  
  
###  Features and Polynomial Regression
  
  
We can improve our features and the form of our hypothesis function in a couple different ways.
  
We can combine multiple features into one. For example, we can combine x_1 and x_2 into a new feature x_3 = x_1 * x_2.
  
**Polynomial Regression**
  
Our hypothesis function need not be linear (a straight line) if that does not fit the data well.
  
We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).
  
For example, if our hypothesis function is <p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{x}%20=%20&#x5C;theta_0%20+%20&#x5C;theta_1%20x_1%20+%20&#x5C;theta_2%20x_1^2"/></p>  
 then we can create additional features based on x_1, to get the quadratic function or the cubic function <p align="center"><img src="https://latex.codecogs.com/gif.latex?h_&#x5C;theta{x}%20=%20&#x5C;theta_0%20+%20&#x5C;theta_1%20x_1%20+%20&#x5C;theta_2%20x_1^2%20+%20&#x5C;theta_3%20x_1^3"/></p>  
  
  
In the cubic version, we have created new features x_2 and x_3 where <p align="center"><img src="https://latex.codecogs.com/gif.latex?x_2%20=%20x_1^2"/></p>  
  
 and <p align="center"><img src="https://latex.codecogs.com/gif.latex?x_3%20=%20x_1^3"/></p>  
.
  
 **NOTE!!! One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.**
  
 eg. if x_1 has range 1 - 1000 then range of x_1^2 becomes 1 - 1000000 and that of x_1^3 becomes 1 - 1000000000.
  
##  Computing Parameters Analytically
  
  
###  Normal Equation
  
  
Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:
  
![Normal Equation Example](images/NE01.jpg )
  
There is **no need** to do feature scaling with the normal equation.
  
The following is a comparison of gradient descent and the normal equation:
  
| Gradient Descent           | Normal Equation                                              |
| -------------------------- | ------------------------------------------------------------ |
| Need to choose alpha       | No need to choose alpha                                      |
| Needs many iterations      | No need to iterate                                           |
| O (k*n^2)                  | O (n^3), need to calculate inverse of X^T * X                |
| Works well when n is large | Slow if n is very large                                      |
  
With the normal equation, computing the inversion has complexity O(n^3). So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.
  
###  Normal Equation Noninvertibility
  
  
When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of theta even if X^T * X is not invertible(singular/ degenerate).
  
If X^T * X is **noninvertible**, the common causes might be having :
  
* Redundant features, where two features are very **closely related** (i.e. they are linearly dependent)
* Too many features (e.g. m <= n). In this case, delete some features or use "regularization" (to be explained in a later lesson).
  
Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.
  