# 5/1/2025, 6:30:45 PM_Numerical Algorithms for Lasso Problems  

# 0. Numerical Algorithms for Lasso Problems  

# 1. Introduction  

Lasso (Least Absolute Shrinkage and Selection Operator) regression, originally introduced by Robert Tibshirani in 1996 [8], stands as a powerful and widely adopted linear model regularization technique in statistics and machine learning [8,21]. It is primarily recognized for its capacity to perform simultaneous feature selection and model estimation [1,3,11,13,21,27].  

![](images/6814f6fb0e89f73f5794d5a21a2b2c1bc7f8a01d32d90e0103126d9ca2fc4331.jpg)  

By incorporating an L1 norm penalty term, defined as $\lambda$ times the sum of the absolute values of the regression coefficients, into the standard linear regression objective function, Lasso effectively addresses several critical issues in predictive modeling [1,13].​  

A significant advantage of Lasso is its inherent ability to induce sparsity in the model coefficients [1,3,11,13,27]. Unlike Ridge regression which employs an L2 penalty and primarily shrinks coefficients towards zero, Lasso's L1 penalty can shrink the coefficients of less influential features exactly to zero [1,13,26]. This property facilitates automatic feature selection [1,3,11,20,21], leading to simpler, more interpretable models and mitigating the risks of overfitting, particularly in settings where the number of features is large or features are highly correlated [3,5,13,21,22,23]. Lasso is particularly valuable when dealing with high-dimensional data and multicollinearity issues [3,8,20,21].  

Despite its statistical advantages, the practical application of Lasso presents computational challenges. The objective function involves minimizing the sum of squared errors plus the L1 norm of the coefficient vector:​  

$$
\operatorname* { m i n } _ { \beta } \| y - X \beta \| _ { 2 } ^ { 2 } + \lambda \| \beta \| _ { 1 }
$$  

where $y$ is the response vector, $X$ is the design matrix, $\beta$ is the coefficient vector, and $\lambda \geq 0$ is the regularization parameter. The L1 norm, defined as $\| \beta \| _ { 1 } = \sum _ { j = 1 } ^ { p } | \beta _ { j } |$ , is non-differentiable at zero for each component $\beta _ { j }$ [4,25]. This nonsmoothness precludes the direct application of standard gradient-based optimization methods like gradient descent or traditional Newton-Raphson algorithms, which rely on differentiability to find analytical solutions [4,25]. Furthermore, the ever-increasing scale of modern datasets, characterized by both a large number of observations and a high dimensionality (number of features), significantly escalates the computational burden associated with solving Lasso problems [3,8,13]. Efficient algorithms are essential to handle the scale and the non-smooth nature of the optimization problem [1].​  

To address these computational hurdles, a diverse range of numerical algorithms has been developed and refined over the years. The evolution of these methods reflects the ongoing effort to balance computational efficiency, convergence guarantees, and the ability to handle large and complex datasets.​  

This survey aims to provide a comprehensive overview of the numerical algorithms developed for solving Lasso problems. We will first formally define the Lasso optimization problem. Subsequent sections will delve into the details of various algorithmic approaches, analyze their performance characteristics, discuss extensions and variants of the basic Lasso model, explore its wide-ranging applications, and finally, outline potential future directions in this active research area.  

# 2. Lasso Problem: Background, Formulation, and Properties  

The Least Absolute Shrinkage and Selection Operator (Lasso) represents a significant advancement in statistical modeling and signal processing, primarily recognized for its capacity to perform simultaneous coefficient estimation and variable selection. It is fundamentally a linear model designed to estimate sparse parameters [11,18]. The core of the Lasso method lies in its objective function, which typically comprises a standard least squares loss term combined with an L1 norm penalty on the model coefficients [1,3,5,8,11,13,16,18,20,21].​  

The standard mathematical formulation of the Lasso problem in its primal or Lagrangian form is expressed as minimizing the sum of the mean squared error (MSE) and a penalty proportional to the sum of the absolute values of the coefficients [5,20]. For a linear regression model, this objective function is given by:  

$$
\operatorname* { m i n } _ { w } \ \frac { 1 } { 2 n } \| X w - y \| _ { 2 } ^ { 2 } + \lambda \| w \| _ { 1 }
$$  

In this formulation, \(X \in \mathbb{R}^{n \times $ \mathsf { p } \} |$ ) represents the feature matrix for $\left\backslash \left( \mathsf { n } \right\backslash \right)$ samples and \(p\) features, \(y \in \mathbb $\{ \mathsf { R } \} \wedge \mathsf { n } \backslash )$ is the target variable vector, $\backslash ( \mathsf { w } \sin \mathsf { \backslash m a t h b b \{ R \} ^ { \wedge } p } ) \quad$ is the vector of coefficients to be estimated, \(\|Xw - $y \backslash | _ { - } 2 ^ { \land } 2 \backslash )$ is the squared Euclidean norm representing the sum of squared residuals, and  

is the L1 norm of the coefficient vector. The regularization parameter \(\lambda $> 0 \backslash$ ) (or \(\alpha\) in some contexts [1,8,11,18,20,21,22,25]) governs the trade-off between minimizing the training error and promoting sparsity in the coefficient vector [8,21,22,25]. A larger \(\lambda\) increases the penalty, leading to more coefficients being shrunk towards or exactly to zero.  

A defining characteristic of the Lasso is the inclusion of the L1 norm penalty. Unlike the squared L2 norm used in Ridge regression, the L1 norm is non-differentiable at zero [3,13,25]. This non-smoothness is precisely what drives coefficients of less relevant features to become exactly zero, thereby inducing sparsity in the model and performing automatic feature selection [8,11,21,22,23,27]. This property is crucial for handling high-dimensional data where many features may be irrelevant or redundant.  

The Lasso problem can also be formulated in an equivalent constrained form [3,11,22]. This formulation minimizes the sum of squared errors subject to a constraint on the L1 norm of the coefficient vector:  

$$
\operatorname* { m i n } _ { \beta } { \frac { 1 } { 2 } } \| y - X \beta \| _ { 2 } ^ { 2 } \quad { \mathrm { s u b j e c t ~ t o } } \quad \| \beta \| _ { 1 } \leq t
$$  

Here, $\backslash ( { \mathsf { t } } \backslash )$ is a positive hyperparameter that bounds the L1 norm of the coefficients [3]. The Lagrangian and constrained forms are equivalent in the sense that for any solution to the constrained problem with parameter \(t\), there exists a \ (\lambda\) such that it is also a solution to the Lagrangian problem, and vice versa. This equivalence highlights the geometric interpretation of Lasso optimization, where the solution is found on the intersection of the least squares contours and the L1 ball (an origin-centered diamond shape in 2D). The sharp corners of the L1 ball at the axes are where the optimal solution is likely to occur for some components to be exactly zero.  

The non-differentiability of the L1 norm at zero necessitates specialized optimization algorithms. Standard gradient descent cannot be directly applied at points where the derivative is undefined. To address this, the concept of the subdifferential for convex functions is utilized, allowing the extension of convex optimization techniques to the non-smooth Lasso objective [3,13,25]. The subgradient allows for conditions analogous to gradient-based optimality criteria to be defined and used in algorithms.​  

Key properties of the Lasso solution derived from its formulation include sparsity and coefficient shrinkage [13,21,22]. Lasso not only shrinks the magnitude of coefficients but, more importantly, sets some coefficients exactly to zero, effectively excluding the corresponding features from the model. This is in contrast to L2 regularization (Ridge regression), which adds a penalty proportional to the sum of the squared coefficients \(\|\theta\|_2^2\) [5,22,25]. While L2 regularization shrinks coefficients, it rarely sets them exactly to zero, meaning all features typically remain in the model, albeit with reduced influence [22,23,26]. This fundamental difference makes Lasso suitable for feature selection, whereas Ridge regression is primarily used for preventing overfitting and improving the stability of solutions, particularly when features are highly correlated [11,13,18,20,22,23,26]. L1 regularization often results in sparser models than L2.​  

Building upon the strengths of both L1 and L2 regularization, the Elastic Net was introduced. It combines both penalties in its objective function [8,16,18], typically formulated as:​  

$$
\operatorname* { m i n } _ { w } \ \frac { 1 } { 2 n } \| X w - y \| _ { 2 } ^ { 2 } + \lambda _ { 2 } \| w \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| w \| _ { 1 }
$$  

or equivalently, using a single \(\lambda\) and an \(\alpha\) parameter balancing the mix:  

$$
\operatorname* { m i n } _ { \boldsymbol { w } } \ \frac { 1 } { 2 n } \| \boldsymbol { X } \boldsymbol { w } - \boldsymbol { y } \| _ { 2 } ^ { 2 } + \lambda \left( \frac { 1 - \alpha } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 } + \alpha \| \boldsymbol { w } \| _ { 1 } \right)
$$  

where \(\alpha \in \). Elastic Net inherits the sparsity-inducing property of L1 regularization and the grouping effect and stability of L2 regularization, which is particularly beneficial when dealing with correlated features. When \(\alpha $\scriptstyle = 1 \backslash$ ), Elastic Net reduces to Lasso, and when $\scriptstyle \backslash ( \backslash a \vert { \mathsf { p h a } } = 0 \backslash )$ , it becomes Ridge regression (scaled by \(\lambda\)).  

Beyond these standard forms, variations of Lasso and sparse recovery problems exist, such as those employing non-convex penalties for potentially stronger sparsity or oracle properties [2,6,17], or those adapted for specific data structures like nonnegativity [10] or block sparsity [24]. However, the fundamental Lasso formulation with the L1 norm remains the cornerstone for understanding sparse linear models.  

# 2.1 Mathematical Formulation and Basic Properties  

The Least Absolute Shrinkage and Selection Operator (LASSO) is a prominent technique in statistical modeling and signal processing for simultaneous estimation and variable selection. Its foundational mathematical structure is typically presented in the primal, or Lagrangian, form, which combines a standard least squares loss function with an L1 norm regularization term [8,11,16,18,21,25]. This objective function is formulated as minimizing the sum of the mean squared error (MSE) and a penalty proportional to the sum of the absolute values of the model coefficients [5,20].  

A common representation of the Lasso objective function for linear regression is given by:  

$$
\operatorname* { m i n } _ { w } \ \frac { 1 } { 2 n } \| X w - y \| _ { 2 } ^ { 2 } + \lambda \| w \| _ { 1 }
$$  

where $X$ is the feature matrix (of size $n \times p$ , with $\scriptstyle n$ samples and $p$ features), $y$ is the target variable vector (of size $\scriptstyle n$ ), $w$ is the coefficient vector (of size $p$ ) to be estimated, $\| X w - y \| _ { 2 } ^ { 2 }$ ​ is the squared L2 norm representing the sum of squared errors, $\| w \| _ { 1 } = \sum _ { j = 1 } ^ { p } | w _ { j } |$ is the L1 norm of the coefficient vector, and $\lambda > 0$ (or $\alpha$ in some notations [11,16,18,20]) is the regularization strength parameter [8,21,22,25]. This parameter controls the trade-off between fitting the data well (minimizing the sum of squared errors) and promoting sparsity in the coefficient vector [8,22].  

A defining characteristic of the Lasso formulation is the inclusion of the L1 norm, which is non-differentiable at zero [3,13,25]. This non-smoothness is precisely what encourages some coefficients to be exactly zero, leading to sparsity and performing intrinsic feature selection [8,11,22,23]. Unlike L2 regularization (Ridge regression), which shrinks coefficients towards zero but rarely sets them to zero, L1 regularization effectively removes features from the model [22,23,26]. The nondifferentiability of the objective function necessitates the use of specialized numerical optimization methods, as standard gradient-based techniques are not directly applicable across the entire domain [3,13,25]. Concepts such as subgradients become essential for analyzing and optimizing such non-smooth functions, enabling the extension of convex optimization techniques to the Lasso problem.  

The Lasso problem can also be equivalently expressed in a constrained form [3,22]. This formulation seeks to minimize the sum of squared errors subject to a constraint on the L1 norm of the coefficient vector:​  

$$
\operatorname* { m i n } _ { \beta } { \frac { 1 } { 2 } } \| y - X \beta \| _ { 2 } ^ { 2 } \quad { \mathrm { s u b j e c t ~ t o } } \quad \| \beta \| _ { 1 } \leq t
$$  

Here, $t$ is a hyperparameter that limits the maximum allowable L1 norm of the coefficients [3]. This constrained form is related to the Lagrangian form through the parameter $\lambda$ (or $\alpha$ ) and the constraint bound $t$ ; there exists a one-to-one correspondence between the optimal solutions of the two forms for appropriate choices of $\lambda$ and $t$ . The constrained form highlights the geometric interpretation of Lasso, where the optimization is restricted to an L1 ball.​  

While the provided digests detail the core formulations and the non-differentiability introduced by the L1 norm, they do not extensively cover the theoretical conditions for the existence and uniqueness of Lasso solutions or the specific derivation and interpretation of the Karush-Kuhn-Tucker (KKT) conditions for the Lasso problem. The KKT conditions are crucial for characterizing the properties of optimal solutions for convex optimization problems like Lasso, providing necessary and sufficient conditions for optimality under certain settings. However, an in-depth analysis of these conditions or existence/uniqueness properties based solely on the provided digests is not possible.​  

The standard Lasso formulation serves as a basis for various extensions and related sparse models. For instance, the Elastic Net penalty interpolates between the L1 and L2 norms, formulated as  

$$
P _ { \alpha } ( \beta ) = \frac { ( 1 - \alpha ) } { 2 } \| \beta \| _ { 2 } ^ { 2 } + \alpha \| \beta \| _ { 1 } ,
$$  

where $\alpha = 1$ corresponds to Lasso [16]. Other related formulations include penalties based on the L1/2 norm for sparse logistic regression [17], group sparse penalties like the group Capped- $L _ { 1 }$ regularization [2], or techniques employing the ratio of L1 and L2 norms for sparse signal recovery [14,24]. These variations build upon the fundamental concept of using non-L2 norms for regularization to induce different types of sparsity or incorporate additional structural information.  

# 2.2 Regularization Parameter Selection  

The regularization parameter, denoted as $\lambda$ (or sometimes $\alpha$ ), plays a critical role in Lasso regression. Its value dictates the balance between fitting the training data well and promoting sparsity in the model coefficients [5,21]. This balance is fundamental to managing the bias-variance trade-off: a smaller $\lambda$ allows for a more complex model with potentially lower bias but higher variance, while a larger $\lambda$ leads to a simpler model with sparser coefficients, potentially increasing bias but reducing variance [21,22]. When $\lambda$ is zero, Lasso reduces to ordinary least squares [21]. As $\lambda$ increases, it shrinks more coefficients towards zero, facilitating feature selection and reducing model complexity [21]. However, an excessively large $\lambda$ can oversimplify the model, degrading its predictive accuracy [21]. Consequently, identifying an optimal $\lambda$ is paramount for achieving desirable model performance [5,21].​  

<html><body><table><tr><td>Method</td><td>Principle</td><td>Pros</td><td>Cons</td></tr><tr><td>Cross-Validation (CV)</td><td>Estimate performance on unseen data by splitting the dataset and evaluating models trained on subsets.</td><td>Robust estimate of generalization error; Widely applicable.</td><td>Computationally intensive, especially for large datasets or multiple parameters.</td></tr><tr><td>Information Criteria (AIC/BIC)</td><td>Balance model fit and complexity based on training data.</td><td>Computationally faster than CV.</td><td>Theoretical guarantees or empirical</td></tr><tr><td></td><td></td><td></td><td>performance might differ from CV.</td></tr></table></body></html>  

<html><body><table><tr><td>Algorithm- Specific/Novel</td><td>Methods tailored to specific algorithms (e.g., GNR, Bayesian adaptive).</td><td>Can potentially offer faster selection or leverage specific structures.</td><td>Generalizability and theoretical guarantees may require careful evaluation.</td></tr></table></body></html>  

<html><body><table><tr><td>LassoCVLassoCVLassocVLassocVLassocVLassocVLassocCVLassocVLassoCVLassocVLassocVLassoCVLassocv</td></tr><tr><td>LassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCvX</td></tr><tr><td>LassoCVLassocCVLassoCVLassocVLassoCVLassoCVLassocVLassoCVLassoCVLassocCVLassocVLassoCVLassoCV</td></tr><tr><td>LassoCVLassocVLassoCVLassocVLassoCVLassocVLassocVLassoCVLassocVLassoCVLassocv k -1</td></tr><tr><td>LassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassocCVLassocVLassoCVLassoCVLassocVLassoCVLassoCV</td></tr><tr><td>LassoCVLassocVLassocVLassoCVLassocVLassocVLassocVLassoCVLassocVLassocVLassocVLassocVLassoCVl</td></tr><tr><td>assoCVLassoCVLassocVLassoCVLassoCVLassocVLassocVLassoCVLassocCVLassoCV and are available</td></tr><tr><td>for setting the $\alpha$ (or $\lambda$) parameter， which controls the degree of sparsity</td></tr><tr><td>[11,18]. LassoCVLassoCV is often preferred for high-dimensional datasets with correlated</td></tr><tr><td>variables， while is more computationally efficient when the number of features</td></tr><tr><td>significantly exceeds the number of samples [11]. Common approaches for selecting a single</td></tr><tr><td>$\lambda$ from the CV results include choosing the value that yields the minimum mean</td></tr><tr><td>squared error (LambdaMinMSE or or lambda.min) or a larger value within one standard error</td></tr><tr><td>of theminimum (Lambda1SElambda.1se|1se)to favorsparser models without significant performance loss [16,20].</td></tr><tr><td>While effective,CVcan be computationall intensive,particularly when selecting multiple parameters,such as the two tuningparametersrequiredfortheelasticnet method,which necessitates performing CVonatwo-dimensionalgrid[17].</td></tr></table></body></html>  

LassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLas soLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLa rsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsICLassoLarsIC LassoLarsICLassoLarsICLassoLarsICLassoLarsIC ssoLarsIC\` facilitate selection using AIC and BIC [18].  

Beyond standard CV and information criteria, researchers have explored novel or algorithm-specific parameter selection methods. For instance, a novel parameter selection method has been developed in the context of high-dimensional Lasso regression using a generalized Newton-Raphson algorithm [4]. Another approach avoids manual parameter setting altogether by adopting a Bayesian framework, which iteratively estimates signal parameters [7]. In this Bayesian context, hyperparameters, such as $\gamma _ { q }$ ​ controlling block sparsity, are adaptively updated during the iterative process, with many $\gamma _ { q }$ ​ values naturally converging to zero [7]. This iterative estimation approach offers an alternative to grid search or single-pass selection methods.  

In summary, selecting the regularization parameter $\lambda$ is crucial for balancing model performance and sparsity. Crossvalidation remains a standard and robust technique, offering good performance estimation by minimizing prediction error, albeit with potentially high computational cost, especially for multi-parameter models or large datasets. Information criteria like AIC and BIC provide computationally faster alternatives, though their theoretical guarantees or empirical performance might differ from CV depending on the specific context. Novel or algorithm-specific methods, such as iterative Bayesian estimation, offer promising avenues to automate parameter selection and potentially improve efficiency or incorporate problem-specific structures, although their generalizability and detailed theoretical guarantees often require careful evaluation. The choice among these methods involves trade-offs between accuracy, computational expense, and theoretical justifications.  

# 3. Numerical Algorithms for Solving Lasso Problems  

Solving the Lasso optimization problem presents a unique challenge due to the non-differentiable nature of the L1 norm regularization term [13]. This characteristic precludes the direct application of standard gradient-based methods that rely on differentiability across the entire domain [13]. Consequently, specialized numerical algorithms are required to effectively find solutions that balance data fidelity with sparsity promotion.  

![](images/7897cb671825654abde199e2b270d995b941e562bf1a514d9e545dc3cf6e9c34.jpg)  

This section provides a comprehensive overview of the primary algorithmic approaches developed for solving Lasso problems and their variants.  

The discussion is organized based on the underlying optimization principles employed by these algorithms, covering major classes such as decomposition methods, first-order methods, splitting methods, and path-following techniques. Each subsequent subsection will delve into a specific algorithmic class, detailing its fundamental concepts, iterative working mechanism, and necessary mathematical derivations or pseudocode for clarity [8].  

Key algorithmic frameworks explored include Coordinate Descent (CD) [1,11,13,16,21,25], which tackles the problem by optimizing one variable at a time while holding others fixed, effectively handling the non-smoothness via univariate softthresholding [3]. Proximal Gradient Methods, including accelerated variants like FISTA, approach the problem by combining a gradient step on the smooth loss function with a proximal step that incorporates the non-smooth L1 penalty through the soft-thresholding operator [19]. The Alternating Direction Method of Multipliers (ADMM) provides a framework for splitting the objective function and constraints, facilitating the solution of coupled problems through iterative updates involving primal and dual variables and utilizing the L1 proximal operator [14,16]. Path-following algorithms like Least Angle Regression (LARS) offer an efficient way to compute the entire Lasso solution path by iteratively adding variables based on their correlation with the residual [1,11,13,21,25]. Beyond these, other methods such as Active Set methods, Interior-Point methods, Generalized Newton-Raphson (GNR) algorithms [4], Majorize-Minimization (MM) based approaches [6], and specialized or hybrid algorithms for problem variants like nonnegative Lasso [7,10] are also discussed.  

A critical aspect covered is the analysis of the computational complexity and convergence properties of these algorithms [8]. The complexity is typically examined in terms of key factors such as the number of samples $( n )$ , the number of variables ( $p$ ), the sparsity level, and the number of iterations required for convergence [9,13]. Convergence analysis focuses on theoretical guarantees, convergence rates (e.g., linear, superlinear, or specific rates dependent on matrix properties), and the conditions under which convergence is guaranteed [4,9,10].  

The section systematically compares the strengths and weaknesses of different algorithms, considering factors that affect their practical performance and scalability. This comparison aims to guide readers in selecting the most suitable algorithm based on the specific characteristics of their problem, such as the density or sparsity of the data matrix, the relationship between $n$ and $p$ (e.g., $n \gg p$ or $p \gg n$ ), and the overall scale of the problem [8]. The discussion highlights that the choice of algorithm often involves trade-offs between computational cost per iteration, total number of iterations to reach desired accuracy, robustness to data properties (like correlation), and ease of implementation. The diverse array of algorithms reflects ongoing efforts to develop more efficient, robust, and scalable methods for Lasso and related sparse learning problems.  

# 3.1 Coordinate Descent Methods  

Coordinate Descent (CD) is an iterative optimization algorithm well-suited for problems involving convex objective functions with separable structures, such as the Lasso problem [1,13]. The core idea is to iteratively minimize the objective function along each coordinate axis while keeping all other coordinates fixed [1,13,25]. This differs from gradient descent, which searches along the negative gradient direction [13]. The mathematical basis for applying CD to convex functions is that if a differentiable convex function is minimized along each coordinate axis, the resulting point is a global minimum [13]. Although the Lasso objective contains a non-differentiable L1 term, its separable structure makes it amenable to coordinatewise minimization using specialized operators.  

![](images/97aa368e38dc271fb1a567580e5e3d2c70fd0bdd1aeac41e8af7a9c71f2dafc4.jpg)  

The specific steps involved in applying coordinate descent to the Lasso problem typically begin with initializing the weight coefficients, often to a zero vector [1,3,25]. The algorithm then enters an iterative phase where each weight coefficient is updated in turn while the others are held constant at their most recent values [1,13,25]. For the $j$ -th coefficient $\beta _ { j }$ , the optimization problem reduces to minimizing the objective function with respect to $\beta _ { j }$ alone:​  

$$
\operatorname* { m i n } _ { \beta _ { j } } \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \biggl ( y _ { i } - \sum _ { k \ne j } X _ { i k } \beta _ { k } - X _ { i j } \beta _ { j } \biggr ) ^ { 2 } + \lambda | \beta _ { j } |
$$  

This univariate minimization problem has a closed-form solution given by the soft-thresholding operator [3]. The update formula for $\beta _ { j }$ ​ is expressed as:  

$$
\beta _ { j } = S \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i j } \Big ( y _ { i } - \sum _ { k \neq j } X _ { i k } \beta _ { k } \Big ) , \lambda \Big )
$$  

where  

$$
S ( z , \lambda ) = \operatorname { s g n } ( z ) { \Big ( } | z | - \lambda { \Big ) } _ { + }
$$  

is the soft-thresholding operator [3]. The term  

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } X _ { i j } \left( y _ { i } - \sum _ { k \neq j } X _ { i k } { \beta _ { k } } \right)
$$  

represents the negative gradient of the smooth part of the objective with respect to $\beta _ { j }$ ​ (evaluated at the current values of the other coefficients) plus the scaled $j$ -th feature, often called the "shrunken residual" or "correlation." The algorithm iteratively updates each coefficient using this soft-thresholding rule until a convergence criterion is met [3].  

Convergence is typically checked by monitoring the change in the coefficient vector across iterations [1,13,25]. The algorithm terminates when the changes in all weight coefficients are sufficiently small or when a preset maximum number of iterations is reached [1,13,25]. For convex problems like Lasso, iterating through all coordinates in a fixed order guarantees convergence to the global minimum. The effect of different coordinate selection strategies (e.g., cyclic, random) can impact the speed of convergence in practice, although cyclic updates are common and straightforward.​  

Coordinate descent offers several advantages for solving Lasso problems. It is conceptually simple and relatively easy to implement. Crucially, it is very efficient, particularly for large-scale problems with sparse data, and is considered one of the fastest methods for Lasso regression [1,11]. The cost per iteration is low as it only involves operations related to a single feature. However, potential disadvantages of coordinate descent can include slow convergence in some cases, especially if variables are highly correlated, and sensitivity to the ordering of coordinates, though this latter point is less critical for convex problems with cyclic updates.​  

Coordinate descent finds application in various settings beyond standard linear regression with an L1 penalty. For instance, it has been successfully applied to sparse logistic regression, even when incorporating non-convex penalties. One study developed a coordinate descent algorithm for sparse logistic regression with $\mathsf { L 1 } / 2$ regularization, relevant for highdimensional biological data [17]. This application requires a specific univariate half-thresholding operator, distinct from the soft-thresholding operator used for L1, which is given by:  

$$
\begin{array} { r } { \beta _ { j } = \mathrm { H a l f } ( \omega _ { j } , \lambda ) = \left\{ \begin{array} { l l } { \frac { 2 } { 3 } \omega _ { j } \Big ( \cos \big ( \varphi ( \lambda , \omega ) \big ) \Big ) ^ { 2 } , } & { \mathrm { i f ~ } | \omega _ { j } | > \frac { 3 } { 4 } \lambda ^ { 2 / 3 } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$  

where $\omega _ { j }$ is related to the current residual and feature $X _ { i j }$ ​ , and $\varphi ( \lambda , \omega )$ satisfies  

$$
\cos { \left( \varphi ( \lambda , \omega ) \right) } = \frac { \lambda } { 8 } \Big ( \frac { \lambda } { \omega } \Big ) ^ { - 3 / 2 } .
$$  

This demonstrates the adaptability of the CD framework to different objective functions by deriving the appropriate univariate minimization step. Coordinate descent, particularly based on the work by Friedman, Hastie, and Tibshirani, is the default algorithm for Lasso implementation in popular libraries like scikit-learn [16,18].  

# 3.2 Proximal Gradient Methods  

The proximal gradient method represents a fundamental algorithmic framework for solving optimization problems involving the sum of a smooth convex function and a non‐smooth convex function, a structure characteristic of the Lasso problem. The Lasso objective function,​  

$$
\operatorname* { m i n } _ { x } \frac { 1 } { 2 } \| A x - b \| _ { 2 } ^ { 2 } + \lambda \| x \| _ { 1 } ,
$$  

comprises a smooth least-squares term and a non‐smooth L1‐norm penalty. The L1 norm is crucial for promoting sparsity in the solution vector $\backslash ( \times \backslash )$ , while its non-differentiability at zero necessitates specialized optimization techniques like proximal gradient methods.  

The core idea of the proximal gradient method is to handle the smooth part with a standard gradient step and the non smooth part with a proximal operator.  

![](images/527f13f955b305a3b8a172377d1fd3c3dc9defabef5f2da055109d921b473622.jpg)  

At each iteration $\left\backslash ( \boldsymbol { \mathsf { k } } \right\backslash )$ , the method performs a gradient descent step on the smooth term at the current iterate $\left\backslash \left( \mathsf { x } ^ { \wedge } \mathsf { k } \right\backslash \right)$ , followed by applying the proximal operator of the non‐smooth term:​  

$$
\begin{array} { r } { \boldsymbol { x } ^ { k + 1 } = \operatorname { p r o x } _ { \alpha _ { k } g } \Big ( \boldsymbol { x } ^ { k } - \alpha _ { k } \boldsymbol { \nabla } f ( \boldsymbol { x } ^ { k } ) \Big ) , } \end{array}
$$  

where  

$$
f ( \boldsymbol { x } ) = \frac { 1 } { 2 } \| \boldsymbol { A } \boldsymbol { x } - \boldsymbol { b } \| _ { 2 } ^ { 2 } , \quad g ( \boldsymbol { x } ) = \lambda \| \boldsymbol { x } \| _ { 1 } , \quad \nabla f ( \boldsymbol { x } ) = \boldsymbol { A } ^ { T } ( \boldsymbol { A } \boldsymbol { x } - \boldsymbol { b } ) ,
$$  

and $\backslash ( \backslash a \vert \mathsf { p h a \_ k } > 0 \backslash )$ is the step size. The proximal operator for a function $\mathsf { \backslash } ( \mathsf { \backslash p h i } ( \mathsf { x } ) \backslash )$ with parameter $\left\backslash \left( \left\backslash \mathsf { t a u } > 0 \right\backslash \right) \right.$ is defined as:  

$$
\mathrm { p r o x } _ { \tau \phi } \big ( v \big ) = \arg \operatorname* { m i n } _ { x } \Big ( \phi ( x ) + \frac { 1 } { 2 \tau } \| x - v \| _ { 2 } ^ { 2 } \Big ) .
$$  

For the L1 norm penalty $\backslash ( \mathsf { g } ( \mathsf { x } ) = \backslash \lfloor \mathsf { a m b d a } \ \backslash \left| \mathsf { x } \right\backslash \left| \mathsf { \_ { - } } 1 \right\backslash )$ , the proximal operator \(\operatorname{prox}_{\alpha \lambda \|\cdot\|_1} $( \mathsf { v } ) \backslash )$ has a well-known analytical solution, which is the soft-thresholding operator. For each component $\mathsf { \backslash } ( \mathsf { v \_ i } )$ of the vector \ $( \mathsf { v } \backslash )$ , the soft-thresholding operation is given by:  

# 无效公式  

This operator effectively shrinks the components of $\mathsf { \backslash } ( \mathsf { v } \backslash )$ towards zero and sets components with magnitude less than \ (\alpha \lambda\) exactly to zero, directly implementing the sparsity-promoting effect of the L1 norm. The existence of this efficient analytical solution for the L1 proximal operator is a key advantage that makes proximal gradient methods highly suitable for Lasso. While the L1 norm's proximal operator is straightforward, computing proximal operators for other sparsity-inducing norms can be more involved. For example, for the L1/L2 functional an analytical solution for its proximal operator has been derived, and addressing the unknown signal sparsity challenge during computation may involve methods like a bisection search [14].  

The choice of the step size \(\alpha_k\) significantly impacts the convergence speed and stability of proximal gradient methods. Common strategies include using a fixed step size (typically \(\alpha \le 1/L\), where $\backslash ( \mathsf { L } )$ is the Lipschitz constant of \(\nabla f\)), or using backtracking line search to adaptively determine $\mathsf { \backslash } ( \mathsf { \backslash a l p h a \_ k \backslash } )$ at each iteration. A fixed step size based on the Lipschitz constant guarantees convergence but can be slow, while backtracking line search can improve practical performance by allowing larger step sizes when possible.​  

To further accelerate convergence, variants like the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) have been developed. FISTA, proposed by Beck and Teboulle, builds upon the proximal gradient method by incorporating a momentum term, leading to a faster convergence rate. Specifically, FISTA achieves an $\backslash ( \mathsf { O } ( 1 / \mathsf { k } ^ { \wedge } 2 ) \backslash )$ convergence rate for the objective function value, compared to the $\mathsf { \Omega } ( 0 ( 1 / \mathsf { k } ) \backslash )$ rate of the basic proximal gradient method for general convex problems. This acceleration is achieved by computing the gradient at an extrapolated point rather than the current iterate. FISTA has been applied effectively to large-scale LASSO problems [19], and further refinements—such as the iterative reduction FISTA algorithm—have been proposed to enhance the convergence rate specifically for large-scale instances of LASSO [19].​  

The convergence properties of proximal gradient methods and their accelerated variants like FISTA are well-established for convex functions. For the Lasso problem, given its convex structure, these methods are guaranteed to converge to the optimal solution under appropriate conditions on the step size. The rate of convergence depends on whether acceleration techniques are used and the specific properties of the objective function.​  

# 3.3 Alternating Direction Method of Multipliers (ADMM)  

The Alternating Direction Method of Multipliers (ADMM) is a powerful optimization algorithm particularly well-suited for problems involving composite objective functions, a characteristic often found in regularized regression problems like Lasso. Its core principle lies in splitting variables to decouple the objective function and constraint, allowing for easier minimization steps [16].​  

The standard Lasso problem, minimizing  

$$
\frac { 1 } { 2 } \| A x - b \| _ { 2 } ^ { 2 } + \lambda \| x \| _ { 1 } ,
$$  

can be cast into the ADMM framework by introducing an auxiliary variable $ { \backslash } ( z  { \backslash } )$ . The problem is reformulated as minimizing​  

$$
l ( \boldsymbol { x } ) + g ( \boldsymbol { z } )
$$  

subject to  

$$
A x + B z = c .
$$  

For Lasso, this takes the form:  

Minimize  

$$
\frac { 1 } { 2 } \| A x - b \| _ { 2 } ^ { 2 } + \lambda \| z \| _ { 1 } ,
$$  

subject to  

$$
x - z = 0
$$  

[16].  

Here,  

$$
l ( x ) = \frac { 1 } { 2 } \| A x - b \| _ { 2 } ^ { 2 }
$$  

corresponds to the data fidelity term, and  

$$
g ( z ) = \lambda \| z \| _ { 1 }
$$  

corresponds to the L1 regularization term. The constraint $1 ( x - z = 0 1 )$ enforces the equality between the original variable an the auxiliary variable.  

ADMM tackles this constrained optimization problem by optimizing an augmented Lagrangian function. The iterative process involves updating the primal variables $( \backslash ( \times \backslash )$ and $\left\backslash \left( z \backslash \right) \right\}$ and a dual variable $( \backslash ( { \mathsf { u } } \backslash ) .$ , related to the constraint). The iterative update rules for the Lasso problem, particularly when operating on tall arrays as described in [16], are given by:  

$$
\begin{array} { r l } & { { \boldsymbol x } ^ { k + 1 } = ( { \boldsymbol A } ^ { T } { \boldsymbol A } + \rho { \boldsymbol I } ) ^ { - 1 } \Big ( { \boldsymbol A } ^ { T } { \boldsymbol b } + \rho \big ( { \boldsymbol z } ^ { k } - { \boldsymbol u } ^ { k } \big ) \Big ) } \\ & { { \boldsymbol z } ^ { k + 1 } = S _ { \lambda / \rho } \Big ( { \boldsymbol x } ^ { k + 1 } + { \boldsymbol u } ^ { k } \Big ) } \\ & { { \boldsymbol u } ^ { k + 1 } = { \boldsymbol u } ^ { k } + { \boldsymbol x } ^ { k + 1 } - { \boldsymbol z } ^ { k + 1 } } \end{array}
$$  

where $\left\backslash ( \boldsymbol { \mathsf { k } } \right\backslash )$ is the iteration number, $\mathsf { \backslash } ( \mathsf { A } \backslash )$ is the data matrix, $\left\backslash ( \mathsf { b } \backslash ) \right.$ is the response vector, $\left\backslash \left( \backslash \mathsf { h o } > 0 \right\backslash \right)$ is the penalty parameter, and $\backslash ( \mathsf { S \_ } \{ \backslash \mathsf { k a p p a } \} ( \backslash \mathsf { c d o t } ) \backslash )$ is the soft thresholding operator defined as:​  

$$
S _ { \kappa } ( a ) = \left\{ \begin{array} { l l } { a - \kappa , } & { a > \kappa , } \\ { 0 , } & { | a | \leq \kappa , } \\ { a + \kappa , } & { a < - \kappa } \end{array} \right.
$$  

[16].​  

The $\backslash ( \times \backslash )$ -update typically involves solving a linear system, while the $ { \backslash } ( z  { \backslash } )$ -update involves applying the soft thresholding operator element-wise, which is the proximal operator of the L1 norm. The $\backslash ( \mu \backslash )$ -update is a simple dual ascent step that encourages the constraint $1 ( x - z = 0 1 )$ to be satisfied.​  

Regarding convergence properties, ADMM for convex problems like Lasso is known to converge globally under mild assumptions [14]. Under suitable conditions, a linear convergence rate can be achieved [14]. The choice of the penalty parameter \(\rho\) significantly influences the convergence speed and, potentially, the solution accuracy (though ADMM converges to the correct solution as long as $\backslash ( \backslash \mathsf { r h o } > 0 \backslash )$ ). Larger values of \(\rho\) tend to emphasize constraint satisfaction more strongly in early iterations, potentially leading to faster convergence in some cases but can also make the subproblems harder to solve or cause oscillations. Smaller values might lead to slower convergence but potentially smoother progress. Optimal parameter selection is problem-dependent and often requires tuning or more advanced adaptive methods, though specific guidelines are not detailed in the provided digests.​  

ADMM demonstrates flexibility in handling variations of the Lasso problem or related penalties. For instance, a specific variable-splitting scheme within the ADMM framework has been optimized for the L1/L2 norm minimization problem [14]. This adaptation highlights ADMM's ability to accommodate different non-smooth regularization terms by designing appropriate splitting schemes and corresponding proximal steps.  

# 3.4 Least Angle Regression (LARS)  

Least Angle Regression (LARS) is a notable algorithm frequently employed for solving Lasso regression problems, particularly advantageous in high-dimensional settings where the number of features $( n )$ significantly exceeds the number of samples $( m )$ [11,13,21]. LARS shares conceptual similarities with forward selection and forward stagewise regression methods [13].​  

The core principle of LARS involves iteratively selecting features based on their correlation with the current residual and moving the coefficient estimates in a direction that is equiangular with the active predictors [1]. The process begins by initializing all weight coefficients to zero, making the initial residual vector equal to the target label vector $y$ [1]. In each step, LARS identifies the feature most correlated with the current residual [1,11]. It then updates the coefficient of this feature, moving along its direction, until another feature's correlation with the residual becomes equal to that of the currently active feature [1]. At this point, the algorithm adjusts the coefficients of al active features simultaneously so that the residual vector remains on the bisector of the angles between the active features, ensuring an equiangular path with respect to these features [1]. This iterative procedure continues, adding a new feature to the active set in each step, until the residual is sufficiently small or all features have been incorporated into the model [1,11].  

A key modification makes LARS equivalent to Lasso: while LARS continues adding features as long as their correlation with the residual is highest, the Lasso variant of LARS specifically halts the progress of a coefficient (and potentially removes it from the active set) if its value reaches zero [1]. This modification allows LARS to efficiently compute the entire Lasso regularization path, which is piecewise linear [11]. The ability to generate the full path is computationally efficient and highly beneficial for tasks like model cross-validation [11].  

LARS offers significant computational advantages, especially in high-dimensional scenarios ( $n \gg m$ ), where it can be more efficient than methods that require solving quadratic programs at each step [11,13]. However, a notable drawback of LARS is its potential instability, particularly in the presence of highly correlated features [1]. It can also be sensitive to noise in the samples [11].​  

# 3.5 Other Algorithms and Hybrid Approaches  

Beyond widely discussed iterative methods like coordinate descent and proximal gradient algorithms, several other numerical approaches address the Lasso optimization problem, each with distinct principles and computational characteristics. Active Set methods represent a class of algorithms that iteratively identify and optimize over a subset of features believed to have nonzero coefficients in the solution. The core principle involves starting with an empty or small active set, iteratively adding variables that violate the Karush–Kuhn–Tucker (KKT) conditions, optimizing the unconstrained least squares problem restricted to the current active set, and removing variables that become zero or negative (in the nonnegative case) during the restricted optimization. Preliminary algorithms such as forward selection and forward stagewise regression exemplify the greedy selection principle fundamental to active set approaches, iteratively adding variables most correlated with the current residual [13]. Similarly, Basis Pursuit (BP), a related sparse recovery method, employs a greedy strategy by iteratively selecting relevant basis vectors based on correlation [12]. While Active Set methods can be efficient when the true active set is small or changes slowly, the cost of identifying the correct active set can become substantial in high-dimensional settings.​  

Interior-Point methods offer a different perspective by reformulating the Lasso problem as a convex quadratic program with linear inequality constraints. These methods traverse the interior of the feasible region, converging towards the optimal solution on the boundary. They typically rely on Newton-type steps applied to a barrier function approximation of the constrained problem. While providing strong theoretical convergence guarantees, including polynomial time complexity, standard Interior-Point methods can incur high computational costs per iteration, primarily due to the need to solve large linear systems, which can be prohibitive for very high-dimensional datasets. The provided digests do not delve into specific implementations or complexities of Interior-Point methods for Lasso.​  

Generalized Newton–Raphson (GNR) algorithms represent another class of methods, adapting Newton’s method for nonsmooth optimization problems like Lasso. The GNR algorithm is derived directly from the KKT conditions of the Lasso problem, utilizing generalized Newton derivatives to handle the nondifferentiability of the $L _ { 1 }$ ​ norm at zero [4]. This results in a nonsmooth Newton-type iteration that, under certain conditions, can exhibit fast convergence rates (potentially quadratic) in the vicinity of the solution [4]. A related algorithm, SDAR, is also motivated by the KKT conditions and generalized Newton derivatives, although its original application discussed in one digest is for $L _ { 0 }$ ​ -penalized problems rather than Lasso [9]. The computational expense of GNR depends heavily on the ability to efficiently compute and invert the generalized Hessian or solve the related linear system at each step, which can be challenging in high dimensions.​  

Majorize–Minimization (MM) algorithms provide a flexible framework for optimizing complex objective functions by iteratively minimizing a sequence of simpler surrogate functions that majorize the original objective. The core idea is to replace the original function with an upper bound (the majorizing function) that is easier to optimize. Minimizing the surrogate function provides an update step for the variable, guaranteeing a decrease in the original objective value [6]. This process is repeated until convergence. MM-based algorithms can be particularly useful for nonconvex or composite objectives, transforming them into sequences of convex or unconstrained problems. The provided digest highlights the leveraging of the MM technique for sparse signal recovery, illustrating its applicability in related fields [6].​  

Specialized algorithms are also developed for variants of the Lasso problem, such as nonnegative Lasso, where the coefficients are constrained to be nonnegative. An example is the ReLU-based Hard-thresholding algorithm (RHT) [10]. RHT incorporates the Rectified Linear Unit (ReLU) function within a hard-thresholding framework to explicitly enforce the nonnegativity constraint, combining shrinkage with projection onto the nonnegative orthant [10]. This approach belongs to the class of hard-thresholding algorithms, which perform variable selection by simply setting coefficients below a certain magnitude to zero. RHT has been compared against various other methods for nonnegative sparse signal recovery, including Nonnegative Least Squares (NNLS), standard Iterative Hard-Thresholding (IHT), Hard-Thresholding Pursuit (HTP), and Newton-step–based variants like NSIHT and NSHTP, as well as Nonnegative Orthogonal Matching Pursuit (NNOMP) and its derivatives (FNNOMP, SNNOMP) [10].​  

Analyzing the advantages and disadvantages reveals a tradeoff between computational cost per iteration, convergence speed, and global optimality guarantees. Greedy/Active Set methods can be fast per step but lack global guarantees, and their efficiency depends on the active set size. Newton-type methods like GNR offer potentially fast local convergence but are computationally expensive per iteration. MM methods are robust and flexible, but their performance depends on the surrogate function choice. Specialized methods like RHT are tailored for specific problem variants, offering efficiency for that variant but limited generality.  

Hybrid approaches combine elements from different algorithmic classes to leverage their respective strengths. For instance, combining adaptive Lasso priors with block-sparse Bayesian learning integrates Lasso’s $L _ { 1 }$ ​ sparsity with Bayesian methods’ ability to model structural dependencies (such as block sparsity) and incorporate prior information [7]. Such hybrid models may also utilize techniques like overcomplete dictionaries to enhance signal representation and recovery. Hybridization can lead to more robust, flexible, or efficient algorithms, potentially overcoming the limitations of single methods by integrating different optimization paradigms or problem formulations.  

# 3.6 Algorithmic Convergence and Complexity Analysis  

The analysis of algorithmic convergence and computational complexity is crucial for understanding the performance and scalability of numerical methods applied to Lasso problems. Different algorithms offer varying theoretical guarantees on convergence and exhibit distinct computational profiles influenced by problem dimensions, sparsity, and specific algorithmic design.​  

<html><body><table><tr><td>Algorithm(s)</td><td>Convergence Property</td><td>Complexity per Iteration</td><td>Key Factor(s) Rate/Complexity</td></tr><tr><td>Generalized Newton- Raphson (GNR)</td><td>Local one-step convergence</td><td>High (solving linear system)</td><td>Proximity to solution, matrix properties</td></tr><tr><td>ADMM</td><td>Global convergence, potentially linear rate</td><td>Varies (often solving linear system)</td><td>Penalty parameter p, matrix properties</td></tr><tr><td>SDAR</td><td>Specific bounds on error decay(l2,lo）</td><td>O(np)</td><td>Sparsity (J), coefficient magnitude (R), matrix conditions (Riesz, coherence)</td></tr><tr><td>Coordinate Descent /Gradient Descent</td><td>Typically sublinear (0(1/k) or 0(1/k^2) with acceleration)</td><td>O(mn) (m samples,n dims)</td><td>Data structure, correlation</td></tr><tr><td>Hard-Thresholding (e.g., RHT)</td><td>Stable recovery conditions</td><td>Varies (often matrix products)</td><td>Matrix properties (RIP, coherence)</td></tr></table></body></html>  

Regarding convergence properties, several theoretical guarantees have been established for various Lasso algorithms. For instance, a Generalized Newton-Raphson (GNR) algorithm has been shown to achieve local one-step convergence, indicating rapid convergence once the algorithm is sufficiently close to the solution [4]. In contrast, an Alternating Direction Method of Minimization (ADMM) scheme for related sparse signal recovery problems is proven to have global convergence under mild assumptions, further demonstrating a linear convergence rate under specific conditions [14].  

Convergence speed and the nature of the solution are often tied to the characteristics of the sensing or design matrix. For hard-thresholding–type algorithms, stable recovery conditions can be established based on properties like the Restricted Isometry Property (RIP) and mutual coherence of the sensing matrix. These conditions are posited to be the most favorable for ensuring stable recovery [10]. For algorithms like SDAR, convergence rates are expressed in terms of estimation error decay under specific matrix conditions. Under a sparse Riesz condition on the design matrix, the $\boldsymbol { \ell } _ { 2 }$ ​ estimation error of the solution sequence decreases exponentially to the minimax error bound within $O ( { \sqrt { J } } \log ( R ) )$ steps with high probability, where $J$ is the number of important predictors and $R$ is the relative magnitude of nonzero coefficients. Alternatively, under a mutual coherence condition, the $\ell _ { \infty }$ ​ estimation error converges to the optimal bound in $O ( \log ( R ) )$ steps [9]. These results explicitly link the number of iterations required for a certain accuracy level to the sparsity level $J$ and coefficient magnitude $R$ , mediated by matrix properties.  

Computational complexity analysis reveals significant variations across different algorithms. For standard approaches like Coordinate Descent and Gradient Descent, the computational cost per iteration is typically $O ( m n )$ , where $\mid m \mid$ is the number of samples and $n$ is the dimension of the coefficient vector [13]. The SDAR algorithm has a computational cost of $O ( n p )$ per iteration, where $n$ is the number of samples and $p$ is the dimensionality [9]. The complexity of the Lasso fitting process itself can depend on the input format; using a covariance matrix for fitting $N$ data points and $D$ predictors has a rough complexity of $D ^ { 2 }$ , whereas fitting without a covariance matrix is roughly $N \times D$ [16]. More complex algorithms, such as those involving grid-based searches, can have iterative costs that change during execution. For example, one algorithm exhibits a total time complexity per iteration of $O ( Q ^ { 2 } + L Q M ^ { 2 } + Q ^ { 3 } )$ , where $Q$ is the number of grid points. This complexity decreases as the algorithm progresses by eliminating grid points below a certain threshold, thus adapting the computational load iteratively [7].  

The trade-offs between theoretical convergence guarantees and practical computational efficiency are evident. Algorithms with strong local guarantees like one-step convergence [4] may require careful initialization, while globally convergent methods like ADMM [14] provide robustness but might have different per-iteration costs or convergence rates in practice compared to their theoretical best-case scenarios. Similarly, algorithms with favorable iteration counts for achieving specific error bounds, such as SDAR with its dependence on $J$ and $R$ [9], must be evaluated considering their $O ( n p )$ cost per iteration [9] against the total number of iterations needed. The specific structure of the problem (e.g., sparse vs. dense design matrix, specific relationships between $n$ and $p$ ) heavily influences which algorithm's complexity profile is most efficient. While some analyses provide conditions for stable recovery [10] or detailed per-iteration costs [7,9,13], a comprehensive picture of the total computational work required to reach a certain accuracy, encompassing both iteration cost and total iterations, is necessary for practical comparisons. Based on the provided information, explicit discussions on major theoretical gaps or widely recognized open problems in the convergence analysis of specific Lasso algorithms are not available in the selected digests.​  

# 4. Extensions, Modifications, and Advanced Topics  

Building upon the foundation of the standard Lasso framework, significant research efforts have been dedicated to developing extensions, modifications, and advanced techniques to enhance its performance, address specific data structures, and tackle challenges posed by large-scale datasets. This section delves into these developments, providing a comprehensive overview of how the core Lasso concept has been adapted and expanded.​  

One primary direction involves modifying the regularization term to incorporate structural information or improve statistica properties beyond simple sparsity. Variants such as the Elastic Net, which combines L1 and L2 penalties, are explored for their ability to handle multicollinearity and improve stability, particularly when feature dimensions are high [8,16,22,23]. Group Lasso extends sparsity to predefined blocks or groups of variables, encouraging group-level selection [2]. Adaptive Lasso employs data-dependent weights on the L1 penalty, aiming for improved variable selection consistency and potentially reduced estimation bias for large coefficients [7,9,28]. Similarly, Block Sparse Lasso focuses on selecting contiguous or structured blocks of variables, with theoretical conditions studied for successful signal recovery [24]. These variants provide tailored approaches for scenarios with different underlying data structures or desired model properties.  

A parallel avenue of research explores replacing the convex L1 penalty with non-convex alternatives. The motivation stems from the potential of non-convex penalties to better approximate the desirable L0 norm (which counts non-zero elements), leading to solutions with less estimation bias for large coefficients compared to L1 regularization [14]. Examples include the Capped-L1, SCAD, MCP, L1/2, and L1/L2 norm ratio penalties [2,6,14,17]. While offering potential statistical advantages, optimizing problems with non-convex objectives presents significant computational challenges, often involving specialized algorithms designed to handle local minima and convergence properties [2,6,17].​  

Furthermore, scaling Lasso algorithms to handle extremely large datasets is a critical challenge in modern applications. This requires moving beyond traditional in-memory solvers to techniques suitable for data that exceeds memory capacity or is distributed across multiple computing nodes [16]. Approaches like distributed optimization, stochastic gradient descent (SGD), mini-batch methods, and online algorithms are employed. These methods involve trade-offs, particularly regarding communication costs in distributed settings and convergence behavior for streaming data, which are active areas of investigation.​  

Collectively, these extensions, modifications, and advanced algorithmic techniques significantly broaden the scope and efficacy of Lasso-based sparse learning, enabling its application to a wider variety of problems, structures, and scales than  

the original formulation.  

# 4.1 Lasso Variants: Elastic Net, Group, Adaptive, Block Sparse, etc.  

While the standard Lasso encourages sparsity through the L1 penalty, several variants have been developed to address its limitations or to incorporate specific structural information about the variables or tasks.  

A prominent variant is the Elastic Net, which combines the L1 penalty of Lasso with the L2 penalty of Ridge regression [8,16,17,18,22,23,26]. This combination is particularly useful when dealing with strongly correlated predictors or when the number of features exceeds the number of training instances, scenarios where standard Lasso can be unstable [25].  

Comparing L1 and L2 penalties highlights their fundamental difference: L2 regularization employs a smooth quadratic function, while L1 uses a non-smooth absolute value function [3]. Ridge regression, utilizing the L2 penalty, shrinks coefficients but does not perform feature selection, unlike Lasso which drives some coefficients exactly to zero [3,13]. The objective function for Ridge regression includes an L2 penalty term:  

$$
J ( \theta ) = \frac { 1 } { 2 } ( X \theta - Y ) ^ { T } ( X \theta - Y ) + \frac { 1 } { 2 } \alpha \| \theta \| _ { 2 } ^ { 2 }
$$  

and it can be solved using methods like gradient descent or the normal equation [13].  

l1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_rati ol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratiol1_ratio atio\` [18]. The Elastic Net framework has also been extended to develop robust estimators, such as adaptive PENSE, which can handle heavy-tailed error distributions and outliers while possessing the oracle property [28].  

Another significant variant is Group Lasso, designed for problems where variables have a natural group structure, such as dummy variables representing a categorical feature or genes within a biological pathway. Group Lasso encourages sparsity at the group level, meaning that either all coefficients within a group are zeroed out or none are. Research in this area includes exploring relaxations using penalties like the Capped- $\ell _ { 1 }$ ​ penalty to address associated optimization challenges [2].  

Adaptive Lasso is a variant that assigns data-dependent weights to the L1 penalty for each variable. This weighting allows for differential shrinkage and can improve the variable selection consistency property compared to the standard Lasso [7]. The adaptive weights are typically determined from an initial consistent estimator, often obtained via Ridge regression or standard Lasso. The use of adaptive weights aims to mitigate issues where standard Lasso might over-shrink large coefficients or fail to select variables consistently under certain conditions [9]. Adaptive Lasso priors have been utilized in applications like direct localization of radiation sources to improve recovery performance [7].​  

Block Sparse Lasso, related to Group Lasso, specifically addresses signals or coefficients structured in contiguous or predefined blocks. It encourages sparsity by promoting the selection of entire blocks of variables. Analysis of Block Sparse Lasso focuses on the conditions under which block-sparse signals can be accurately recovered, often using techniques like $\ell _ { 2 } / \ell _ { 1 }$ minimization [24]. This type of sparsity is relevant in areas like signal processing where coefficients naturally cluster. Block sparse Bayesian learning methods also exploit such structures and intra-block correlations [7].​  

Beyond these, other variants exist, such as MultiTaskLasso. This variant is applied to multiple regression problems simultaneously, enforcing the constraint that the set of selected features is common across all tasks [18]. This encourages shared sparsity patterns across related problems. While Fused Lasso, mentioned in the subsection description, is another relevant variant for encouraging piecewise constant solutions (not directly found in the digests), the discussed variants – Elastic Net, Group Lasso, Adaptive Lasso, and Block Sparse Lasso – demonstrate how modifications to the L1 penalty or the incorporation of L2 penalties and structural assumptions lead to estimators suited for different data characteristics and modeling objectives. Each variant encourages a distinct type of sparsity or structure, extending the applicability and performance of sparse learning techniques.  

# 4.2 Non-convex Penalties and Optimization  

While the $L _ { 1 }$ ​ penalty promotes sparsity, it is known to potentially introduce bias in the estimation of large coefficients, particularly when predictors are highly correlated. To mitigate this limitation and achieve solutions that better approximate the $L _ { 0 }$ ​ norm (which directly counts nonzero elements), nonconvex penalties have been proposed as alternatives. These penalties aim to retain sparsity while reducing the bias inherent in $L _ { 1 }$ ​ regularization for larger coefficient values. Examples of such nonconvex penalties discussed in the literature include Capped- $L _ { 1 }$ ​ , SCAD, MCP, $L _ { 1 / 2 }$ ​ , and the $L _ { 1 } / L _ { 2 }$ ​ norm ratio ([2,6,14,17]).  

These nonconvex penalties are designed to serve as continuous relaxations of the $L _ { 0 }$ ​ norm, often exhibiting properties   
such as unbiasedness for large coefficients and oracle properties under certain conditions. For instance, the Capped- $L _ { 1 }$   
penalty is explored as a nonconvex relaxation specifically for group sparsity, aiming to provide a better approximation to the   
group $L _ { 0 }$ norm compared to the standard group $L _ { 1 }$ penalty ([2]). Similarly, the $L _ { 1 } / L _ { 2 }$ ​ norm ratio is utilized as a   
nonconvex approach to approximate the $L _ { 0 }$ ​ norm ([14]). The $L _ { 1 / 2 }$ ​ penalty is another well‐studied nonconvex function   
used to induce sparsity ([17]). Research also explores composite functions formed from basic nonconvex forms to achieve a   
deeper approximation of the $L _ { 0 }$ ​ norm, better reflecting the true sparsity characteristics of signals ([6]). For example, a   
composite Delta Approximating (DA) function   
\​   
built from components   
\​   
has been proposed and shown through analysis to potentially offer improved approximation performance compared to its   
constituent parts and other forms like   
\​  

Optimizing problems with nonconvex objectives presents significant challenges compared to convex optimization. These include the potential for multiple local minima, sensitivity to the choice of initialization, and difficulties in establishing global convergence guarantees. Despite these challenges, several algorithms have been adapted or developed to handle nonconvex Lasso-type problems. Common approaches include iteratively reweighted methods (such as iteratively reweighted $L _ { 1 }$ or $L _ { 2 }$ ) and specialized proximal algorithms. For problems involving specific nonconvex penalties, customized algorithms are often necessary. For example, optimizing $L _ { 1 / 2 }$ ​ -penalized logistic regression can be addressed using a coordinate descent algorithm equipped with a specialized univariate half thresholding operator tailored to the $L _ { 1 / 2 }$ function ([17]). While achieving global optimality is often difficult, these algorithms aim to find desirable solutions, such as stationary points, or solutions that exhibit strong sparsity and good estimation properties ([6,17]). The convergence properties of these algorithms are a critical area of study, focusing on conditions under which they converge and the nature of the points they converge to (e.g., local minima, critical points) ([6,17]).  

# 4.3 Large-Scale, Distributed, and Online Lasso  

Traditional numerical algorithms for Lasso problems often face significant challenges when applied to large-scale datasets, particularly when the data size exceeds the available memory or necessitates computation across multiple processing units. Handling such large volumes of data efficiently requires specialized approaches beyond standard in-memory iterative solvers.  

One class of methods designed to address large-scale Lasso problems involves techniques suitable for data that is too large to fit into memory, often referred to as "tall arrays." The Alternating Direction Method of Multipliers (ADMM) has been explored as a framework for solving Lasso problems in such scenarios [16]. ADMM's structure allows for the decomposition of the problem into smaller, more manageable subproblems, which can be beneficial when dealing with computational or memory constraints imposed by large datasets. Specifically, applying ADMM enables solving Lasso problems involving these tall arrays by breaking down the optimization process in a way that mitigates the memory bottleneck [16].  

Beyond memory-constrained scenarios, large-scale problems often benefit from distributed optimization, where the dataset and computation are partitioned across multiple nodes. While distributed approaches are crucial for leveraging parallel processing, they introduce trade-offs, particularly between computational effort on each node and the communication overhead required to synchronize across nodes. Similarly, for datasets too large for a single machine's memory, stochastic gradient descent (SGD) and mini-batch methods are widely used. These iterative techniques update parameters using small subsets of data (mini-batches), reducing the memory footprint per iteration and enabling processing of vast datasets. Furthermore, online Lasso algorithms are designed for streaming data, updating the model incrementally as new data arrives without storing the entire history. While these distributed, SGD/mini-batch, and online methods are significant aspects of solving large-scale Lasso problems, detailed analyses of their application, performance trade-offs (such as computation-communication balance), or specific algorithmic variants were not extensively covered in the provided digest materials.  

# 5. Theoretical Properties and Convergence Analysis  

The theoretical analysis of Lasso and its variants is crucial for understanding their reliability and performance guarantees.  

<html><body><table><tr><td>Concept</td><td>Description</td><td>Importance</td><td>Relevant Matrix</td></tr><tr><td>Consistency</td><td>to true parameters as N increases.</td><td>estimation.</td><td></td></tr><tr><td>Consistency</td><td>identifies true zero/non-zero coefficients as N</td><td>interpretability and feature selection accuracy.</td><td></td></tr><tr><td>Restricted Isometry Property (RIP)</td><td>Matrix acts almost like an isometry on sparse vectors.</td><td>Guarantees stable and accurate recovery of sparse signals.</td><td>RIP constant</td></tr><tr><td>Mutual Coherence</td><td>Maximum correlation between columns of the design matrix.</td><td>Low coherence is desirable for sparse recoveryguarantees.</td><td>Max column correlation</td></tr><tr><td>Oracle Property</td><td>Estimator performs as well as if true sparse support was known in advance.</td><td>Indicates optimal asymptotic performance.</td><td></td></tr></table></body></html>  

A primary focus is on establishing statistical consistency, which refers to the convergence of the Lasso estimator to the true parameters as the data size increases, and variable selection consistency, which ensures that the estimator correctly identifies the true sparse support. These properties are often demonstrated to hold under specific conditions on the design matrix and the true underlying sparse signal [29]. For instance, the Lasso formulation, often expressed as  

$$
\operatorname* { m i n } \| \alpha \| _ { 1 } \quad { \mathrm { s u b j e c t ~ t o } } \quad y = \Phi \alpha + n ,
$$  

promotes sparsity by encouraging coefficients to be exactly zero [12]. This inherent property of L1 regularization, distinct from L2 regularization which tends to shrink coefficients towards zero but rarely exactly to zero, facilitates feature selection and enhances model interpretability [23,26]. From a Bayesian perspective, this behavior is linked to the use of a Laplace prior for L1 regularization, in contrast to the Gaussian prior associated with L2 regularization [26].​  

Key concepts that provide sufficient conditions for accurate signal recovery in sparse settings include the Restricted Isometry Property (RIP) and mutual coherence [10]. The RIP quantifies how close a matrix is to an isometry when restricted to sparse vectors, while mutual coherence measures the maximum correlation between columns of the sensing matrix. Low RIP constants and low mutual coherence are desirable properties for guaranteeing stable and exact recovery, particularly in compressed sensing contexts where the theoretical basis relies on the signal's sparsity and the incoherence between observation and sparse bases [10,29]. Specific theoretical properties have also been established for variants like block sparse recovery. A sharp sufficient condition for block-sparse signal recovery relies on the block Restricted Isometry Property (block-RIP), providing guarantees for stable recovery in the presence of noise and exact recovery in the noise-free scenario [24].​  

Beyond consistency, the oracle property is a highly desirable theoretical characteristic for sparse estimators, implying that the estimator performs asymptotically as well as if the true sparse support were known in advance. Estimators like the $\lfloor 1 / 2$ penalty-based methods have demonstrated attractive properties including unbiasedness, sparsity, and the oracle property [17]. Similarly, the Adaptive PENSE estimator has been shown to possess the oracle property, even under challenging conditions like heavy-tailed error distributions or contamination, without requiring prior knowledge of residual scale or moment conditions [28].​  

The convergence analysis of numerical algorithms employed to solve Lasso problems is another critical area of theoretical study. Different algorithms exhibit varying convergence rates, typically characterized as linear or sublinear, depending on the specific algorithm, the structure of the problem, and assumptions about the data. For example, the ADMM algorithm, when applied to specific L1/L2 norm minimization problems relevant to sparse recovery, has been proven to achieve global convergence with a linear convergence rate under certain conditions [14]. For high-dimensional Lasso regression, generalized Newton-Raphson (GNR) algorithms have demonstrated local one-step convergence properties [4]. These analyses provide guarantees on how quickly an algorithm approaches the optimal solution.​  

Optimality conditions characterize the solutions obtained by numerical algorithms. For convex Lasso problems, KarushKuhn-Tucker (KKT) conditions provide necessary and sufficient conditions for optimality. However, the analysis becomes more complex for problems involving non-convex penalties or structured sparsity. For group sparse problems, particularly those with non-convex relaxations like the Capped- $L _ { 1 }$ ​ penalty, the analysis involves different types of stationary points (Dstationary, C-stationary, and L-stationary) and establishing conditions for the equivalence of solutions between the original sparse problem and its continuous relaxations [2].  

Theoretical challenges persist, particularly concerning non-convex penalties and large-scale settings. While non-convex functions can offer improved approximation performance compared to convex ones [6], leading to potentially better statistical properties like unbiasedness and sparsity [17], their non-convexity complicates the convergence analysis of algorithms and the characterization of global optima [2]. Furthermore, scaling theoretical guarantees and convergence rates to very large datasets and high-dimensional feature spaces poses significant challenges, often requiring assumptions that may not hold in practice or leading to rates that are too slow for practical applications. The bias-variance trade-off is a fundamental concept influencing the theoretical properties and generalization performance of Lasso, where regularization reduces variance but can introduce bias, affecting the overall generalization error [20,22]. Generalization error can be decomposed into bias, variance, and irreducible noise [20]. The Cramer-Rao Lower Bound (CRLB) can also serve as a theoretical benchmark to assess the performance limits of estimators in specific models [7].  

# 无效公式  

# [7]​  

where $\theta$ represents the vector of unknown parameters. Concepts like AIC and BIC provide information criteria related to model complexity and likelihood, offering a means to evaluate model fit in a statistically grounded manner [18].  

# 6. Performance Comparison and Evaluation  

Empirical performance evaluation is crucial for understanding the practical applicability and efficiency of numerical algorithms designed for Lasso problems and their variants [3]. Numerous studies provide numerical examples and simulations to illustrate theoretical results and compare different algorithmic approaches [15]. Common performance metrics employed in the literature include Mean Squared Error (MSE), Root Mean Square Error (RMSE), R-squared $( R ^ { 2 } )$ coefficient, prediction accuracy, reconstruction error, signal-to-noise ratio (SNR), normalized mean square error (NMSE), support set recovery success rate, computational time, and the number of selected variables [5,6,7,10,17,20].  

Comparisons often benchmark Lasso algorithms against classical methods and other regularization techniques. For instance, initial comparisons have shown Lasso potentially outperforming Ordinary Least Squares (OLS) and Ridge regression in terms of MSE on test sets, although performance is highly dependent on the regularization parameter choice [20]. The appropriate tuning of regularization parameters is emphasized for achieving improved model accuracy [22].  

glmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetg lmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglm netglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmnetglmne tglmnetglmnet  package [4]. For non-negative Lasso problems and sparse signal recovery, the ReLU-based Hard  

Thresholding (RHT) algorithm has been extensively evaluated against methods such as NNLS, IHT, HTP, NSIHT, NSHTP, NNOMP, FNNOMP, and SNNOMP, demonstrating competitive recovery performance and efficiency [10].  

LassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCVLassoCV LassoCVLassoCVLassoCVLassoCVLassoCVLassoCV   
large)withsignificantcollinearityamongfeatures, ‘LassoCV ‘(usingcross −   
validation)isoftenpreferred.However, whenthenumberofsamples(   
LassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCV LassoLarsCV\` may offer faster computation [18]. The strategy for handling the covariance matrix can also impact performance; using the covariance matrix might be faster when $N > D$ but potentially less numerically stable [16].  

Beyond standard Lasso, performance evaluations extend to its many variations. The L1/2 penalized sparse logistic regression, evaluated on simulated and real gene expression datasets (Leukaemia, Prostate, Colon, DLBCL), uses average test errors and the number of selected variables to compare its performance against Elastic Net (LEN) and L1 penalized methods [17]. The Symmetric Difference Algorithm for Regression (SDAR), a constructive approach for high-dimensional sparse learning, has shown competitive or superior performance in terms of accuracy and efficiency when compared to Lasso, MCP, and various greedy methods [9]. In the context of robust estimation, the Adaptive PENSE estimator demonstrates superior finite-sample performance compared to other robust regularized estimators when samples are contaminated, while remaining competitive with classical regularized estimators in clean data scenarios [28]. For specific applications like far-field radiation source localization, proposed joint adaptive Lasso and block-sparse Bayesian learning methods have been compared using simulations, evaluating performance with RMSE as a function of SNR, number of snapshots, and antenna elements. These studies indicate that the proposed method outperforms others, particularly unde challenging low SNR or limited snapshot conditions. The RMSE is calculated using the formula:​  

$$
R M S E = \sqrt { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \lVert p _ { n } - p \rVert ^ { 2 } }
$$  

where $p _ { n }$ ​ is the estimated source location and $p$ is the true location [7].  

Studies on non-convex penalties also report empirical benefits. Algorithms based on non-convex composite functions have shown superior performance in reconstruction error, SNR, NMSE, and support set recovery success rate compared to classical algorithms like SL0, IRLS, SCSA, and Basis Pursuit (BP) [6]. Similarly, approaches minimizing the $l _ { 2 } / l _ { 1 }$ ​ norm exhibit efficiency compared to state-of-the-art sparse signal recovery methods in both noisy and noise-free environments [14]. For block-sparse signal recovery, the $l _ { 2 } / l _ { 1 }$ ​ minimization method has been numerically verified, demonstrating robustness and stability [24].​  

The impact of algorithm parameters, such as the regularization parameter $\alpha$ (or $\lambda$ ), on performance is also empirically studied, showing how different values affect metrics like MSE and the resulting model's characteristics, such as curve smoothness in regression [5].​  

LassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLas soLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLa rsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCV LassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLassoLarsCVLass oLarsCV sCV\` when $N \ll D$ . For robust estimation, algorithms like Adaptive PENSE are preferable in the presence of contaminated data. For sparse signal recovery applications, the choice may involve evaluating various thresholding or optimization algorithms based on metrics like reconstruction error and support recovery, potentially favoring non-convex penalties for enhanced sparsity and accuracy under certain conditions.​  

# 7. Applications of Lasso  

The Least Absolute Shrinkage and Selection Operator (Lasso) algorithm and its variants have become indispensable tools across numerous scientific and engineering domains [8,12,18,29].  

![](images/29cbe14b5f5398e23b2fbe8dff77ed32e45345bc0c016718a5f88dd8c9c1bb22.jpg)  

The primary strength of Lasso lies in its ability to promote sparsity in model coefficients or signal representations by incorporating an L1‐norm penalty in the optimization objective. This inherent sparsity-inducing property is leveraged for effective feature selection, the construction of sparse models, and the recovery of sparse signals, particularly in highdimensional settings where identifying relevant variables or components is challenging [11,20,27]. The diverse applications discussed in the following subsections highlight Lasso’s versatility and impact.  

In Signal Processing, Lasso plays a pivotal role in the recovery and analysis of sparse signals. A key application is within Compressed Sensing (CS), where the goal is to reconstruct high-dimensional signals from limited measurements by exploiting sparsity in some domain [29]. Lasso’s L1 regularization drives many coefficients to zero, which is crucial for effectively recovering these sparse signals [10,14]. Variants are used for specific signal structures such as non-negative or block sparsity [7,10,24]. Real-world examples include single-pixel cameras and source localization under challenging conditions like low Signal-to-Noise Ratio (SNR) [7,29]. Performance is typically evaluated using metrics such as SNR and Mean Squared Error (MSE).​  

Within Machine Learning, Lasso is a standard regularization technique used for feature selection, reducing model complexity, and mitigating overfitting [20,22,27]. By setting coefficients to zero, Lasso identifies the most relevant predictors, enhancing model interpretability [11,16,18,20]. It is extensively applied in sparse classification and regression models, improving generalization performance on unseen data [5,18,23]. Performance is commonly assessed using MSE and ${ \mathsf { R } } ^ { 2 }$ score for regression, and accuracy or misclassification error for classification [5,8,17]. Comparisons are often made with related techniques like Elastic Net [17].​  

In Bioinformatics, Lasso is particularly valuable for analyzing high-dimensional biological datasets, such as genomic or proteomic data [20,27]. A key application is identifying a small, relevant subset of genes or biomarkers predictive of disease or outcome [20]. Lasso’s sparsity property is crucial here, performing automatic feature selection by driving irrelevant gene coefficients to zero [20,27]. This leads to more interpretable models and improved prediction ability, especially in challenging scenarios where $p \gg n$ [17,27]. Examples include gene selection for cancer classification [3,10,17].  

The Finance domain also benefits significantly from Lasso, particularly when dealing with datasets containing a large number of potentially correlated features [27]. Lasso’s feature selection capability helps identify truly influential factors, enhancing interpretability and predictive power in tasks like credit risk assessment [27]. It is also useful for selecting sparse asset portfolios [27]. Sparse models built with Lasso tend to exhibit greater stability in noisy financial data compared to dense models.  

In Image Processing, sparse representation techniques, often involving Lasso-like penalties, are utilized for tasks such as denoising, reconstruction, and Compressed Sensing applications [29]. CS-based approaches have been explored for image encryption, hashing, and retrieval [29]. Sparse modeling also finds utility in areas like face recognition and spectral unmixing [10,29].​  

Beyond these major areas, Lasso’s principles extend to various Other Applications. Compressed Sensing principles, closely related to Lasso, are applied in Wireless Sensor Networks (WSN) for efficient data collection and in Magnetic Resonance Imaging (MRI) for accelerated imaging [29]. Linear secure coding also leverages CS for reliable signal recovery [29]. The preference for simpler, sparse models inherent in Lasso resonates with principles like "Occam’s Razor" in management science and philosophy [20].  

Across these diverse fields, Lasso’s core contribution is its ability to navigate high-dimensional data by favoring sparse solutions, leading to models or representations that are often more interpretable, stable, and predictive than their dense counterparts. While challenges exist, such as handling specific data structures or noise levels, the fundamental benefits of sparsity promoted by Lasso make it a widely applicable and powerful tool.​  

# 7.1 Signal Processing  

Lasso plays a pivotal role in various signal processing applications, fundamentally contributing to the recovery and analysis of sparse signals. A primary area of application is sparse signal recovery, particularly within the framework of Compressed Sensing (CS) [29]. In this context, the objective is typically to reconstruct a high-dimensional signal from a limited number of linear measurements, leveraging the prior knowledge that the signal has a sparse representation in some basis. Lasso, by minimizing the sum of squared errors subject to an L1-norm constraint on the coefficient vector, inherently promotes sparsity by driving many coefficients to exactly zero. This property is crucial for effectively recovering sparse signals even from incomplete or noisy data [10,14].​  

The application of Lasso extends to recovering signals with specific structural properties. For instance, algorithms based on Lasso or its variants can address non-negative sparse signal recovery [10] or recover signals exhibiting block sparsity, where non-zero elements appear in contiguous blocks [7,24]. These structural considerations often arise in practical signal processing tasks. A concrete example of CS applied to signal recovery is the single-pixel camera, which reconstructs images from measurements captured using a Digital Micromirror Device and pseudo-random patterns [29].​  

Beyond general signal recovery, Lasso and its adaptations are applied in more specialized signal processing domains. Source localization, for example, can leverage Lasso techniques to estimate the positions of radiation sources, even under challenging conditions characterized by low Signal-to-Noise Ratio (SNR) and limited data availability [7]. While not explicitly detailed in the provided digests, Lasso's sparsity-promoting nature also makes it applicable to signal denoising, where it can identify and retain significant signal components while suppressing noise, and feature extraction, by selecting the most relevant signal components or basis coefficients.​  

The performance of Lasso-based recovery algorithms in signal processing is typically evaluated using standard metrics. As noted in the context of source localization, SNR is a critical factor influencing recovery accuracy, especially in challenging environments [7]. Another widely used metric is the Mean Squared Error (MSE), which quantifies the average squared difference between the recovered signal and the original signal, providing a measure of the reconstruction quality. Addressing challenges like low SNR is essential for robust signal recovery performance [7].​  

# 7.2 Machine Learning  

Lasso (Least Absolute Shrinkage and Selection Operator) has become a standard regularization technique widely applied across various machine learning tasks, particularly for its efficacy in feature selection, reducing model complexity, and mitigating overfitting [20,22,27]. Its application enhances model interpretability by promoting sparse solutions where many coefficients are driven to zero, thereby identifying the most relevant predictors [16,20]. This process of setting coefficients to zero effectively performs feature selection [8,11,18], allowing researchers to identify selected features by examining which coefficients remain non-zero after fitting the model [8].  

Lasso is extensively utilized in both sparse classification and regression models [18]. For regression problems, it has been demonstrated to reduce overfitting, as shown in applications like polynomial regression [5]. Its ability to enhance generalization performance on unseen data is a key benefit [23]. In classification tasks, Lasso, alongside variants like Elastic Net (LEN) and L1/2 penalties, has been applied, for instance, in logistic regression models [17].​  

Performance evaluation of Lasso models in machine learning contexts employs various standard metrics. For regression tasks, common metrics include Mean Squared Error (MSE) [5,8] and R2 score [11]. In classification, evaluation often relies on metrics such as accuracy and misclassification errors [11,17].  

While the subsection description suggests a comparison with techniques like Ridge regression, the provided digests primarily focus on Lasso's properties and comparisons with related L1-based or hybrid penalties. For instance, one study evaluates Lasso against Elastic Net and L1/2 penalized models in classification, using misclassification error as an evaluation metric [17]. This highlights that Lasso is often considered within a broader family of regularization techniques aimed at achieving sparsity and improving model performance.​  

The implementation of Lasso regression is readily available in popular machine learning libraries, such as Scikit-Learn in Python. Practical implementations involve standard steps like data preprocessing (e.g., using StandardScaler), splitting datasets into training and testing sets, fitting the Lasso model to the training data, making predictions on the test set, and evaluating the performance using relevant metrics [8].  

# 7.3 Bioinformatics  

In the field of bioinformatics, Lasso regression serves as a valuable tool, particularly for the analysis of high-dimensional biological data such as genomic or proteomic datasets [20,27]. A key application is the identification of a small, relevant subset of genes or biomarkers that are highly predictive of a specific disease or outcome [20]. This process is crucial for understanding the underlying biological mechanisms and developing diagnostic or prognostic models. For instance, Lasso has been applied to gene selection in cancer classification problems using microarray data, aiming to identify gene biomarkers for distinguishing different cancer types and improving prediction accuracy [10,17].  

Applying statistical models like Lasso in bioinformatics often presents significant challenges, primarily due to the inherent structure of the data. Biological datasets, especially those derived from technologies like microarrays, are typically  

characterized by a very large number of features (e.g., genes) compared to the number of samples (e.g., patients) .  

Despite these challenges, Lasso's $L _ { 1 }$ ​ regularization offers distinct benefits. By promoting sparsity, Lasso effectively performs feature selection, automatically setting the coefficients of irrelevant genes or biomarkers to zero [20]. This results in models that include only a small subset of the most important features, which is highly advantageous in bioinformatics [27]. Sparse solutions enhance the interpretability of the model, making it easier for researchers to identify key biological factors associated with a disease or outcome [27]. Furthermore, selecting only relevant features can improve the prediction ability of the model by reducing noise and preventing overfitting in the ​gpg n scenario [17,27].​  

# 7.4 Finance  

The application of numerical algorithms for solving Lasso problems is particularly valuable in the domain of financial analysis [27]. In this field, researchers and practitioners frequently encounter datasets characterized by a large number of potentially correlated features, making the identification of truly influential factors challenging. Lasso's inherent sparsityinducing property provides a powerful mechanism for feature selection. By driving the coefficients of less relevant features to exactly zero, Lasso effectively identifies key factors from massive feature sets, enhancing the interpretability and predictive capability of financial models [27]. This capability is crucial for tasks such as constructing sparse models for credit risk assessment, where identifying the minimal set of critical variables is essential for both understanding risk drivers and building robust models. Furthermore, Lasso facilitates the selection of a sparse portfolio of assets, an important objective in quantitative finance for managing transaction costs and simplifying portfolio oversight while aiming to achieve desired riskreturn profiles. The benefits of sparsity extend beyond interpretability; in potentially noisy financial data, sparse models built with Lasso tend to exhibit greater stability compared to dense models, as they are less prone to overfitting to irrelevant variations or noise. This improved stability is vital for reliable decision-making in volatile financial markets.​  

# 7.5 Image Processing  

Sparse representation and its associated optimization techniques, such as those based on the Lasso penalty, play a significant role in various image processing tasks. By promoting sparsity, often in a suitable transform domain, these methods facilitate efficient data representation and enable solutions to challenging inverse problems. While standard applications like image denoising and reconstruction from incomplete measurements frequently leverage Lasso for sparsity in domains like wavelets, the principles extend to other areas. Beyond these classical applications, compressed sensing (CS), which heavily relies on sparse recovery techniques akin to Lasso, has been explored in diverse image processing contexts. For instance, CS-based approaches have been investigated for image encryption, image hashing, data hiding, and secure image retrieval, particularly for data transmission over potentially lossy channels where useful information is embedded via CS techniques [29]. Another application area where sparse representation—and thus, potentially Lasso-type methods—finds utility is face recognition [10,29]. These varied applications highlight the versatility of sparse modeling, often underpinned by Lasso or related convex optimization formulations, in addressing complex problems within the field of image processing.​  

# 7.6 Other Applications  

Beyond its prevalent use in statistical modeling and feature selection, Lasso and its underlying principle of sparsity promotion have found application in diverse fields, often leveraging the ability to recover sparse signals or models from limited data. In signal and information processing, applications include Wireless Sensor Networks (WSN), Magnetic Resonance Imaging (MRI), and linear secure coding [29]. For instance, in WSNs, compressed sensing (CS) techniques, closely related to Lasso, facilitate efficient data collection by exploiting the inherent spatiotemporal correlation of sensor data and addressing the resource constraints of sensor nodes by requiring fewer measurements [29]. Similarly, in MRI, CS allows for a significant reduction in the number of measurements needed, which can accelerate imaging procedures and potentially enable real-time MRI capabilities [29]. Linear secure coding also benefits from CS principles, allowing multiple signals to be merged and transmitted with integrated error correction, thereby enabling reliable recovery of the original signals even in the presence of signal loss or damage [29]. Spectral unmixing, a technique used to decompose mixed spectral signals into pure components, is another area where sparsity-promoting methods, often related to Lasso, have been applied [10]. The principle behind Lasso, which favors simpler models by promoting sparsity, also resonates with concepts in other disciplines. In management science, the emphasis on simplicity in models parallels "Occam's Razor," illustrating a convergence of similar ideas across disparate fields, including philosophy [20].  

# 8. Software and Implementations  

The landscape of software and libraries for implementing and solving Lasso problems is diverse, offering researchers and practitioners various tools across different programming languages.  

<html><body><table><tr><td>Library</td><td>Language(s)</td><td>Key Algorithms Implemented</td><td>Notable Features / Use Case</td></tr><tr><td>scikit-learn</td><td>Python</td><td>Coordinate Descent, LARS</td><td>User-friendly ML library, includes CV variants.</td></tr><tr><td>glmnet</td><td>R (Python interface)</td><td>Coordinate Descent</td><td>Efficient for L1-type models,supports various families.</td></tr><tr><td>MATLAB (lasso fn)</td><td>MATLAB</td><td>(Typically Coordinate Descent/Path)</td><td>Integrated function, supports Lasso and Elastic Net.</td></tr><tr><td>NumPy (Custom)</td><td>Python</td><td>Various (CD, Proximal, etc.)</td><td>Flexible for custom algorithm development/resear ch.</td></tr></table></body></html>  

Prominent libraries specifically designed for regularized linear models, including Lasso, are widely utilized.  

scikit-learnscikit-learn library stands out as a popular choice [3,5,8,18,22,27]. Scikitlearn provides the  class for standard Lasso regression [3,5]. It implements algorithms such as coordinate descent and Least Angle Regression (LARS) [1,25]. The library also includes convenient variants for cross-validation, such as LassoCVLassoCVLassoCV and LassoLarsCV rsCV\`, facilitating the selection of the optimal regularization parameter [11]. Examples provided in the literature often demonstrate the ease of use of Scikit-learn for tasks ranging from data preprocessing to model fitting and evaluation [8].  

<html><body><table><tr><td>glmnetglmnetglmnetglmnetglmnet， known for its efficiency and capability in handling L1- type regularized linear models using a coordinate descent algorithm [17]. Primarily</td><td></td><td></td><td></td></tr><tr><td>associated with the R programming language [20]， glmnet also has a Python interface. It</td><td></td><td></td><td></td></tr><tr><td>supports various model families through its family parameter， including Gaussian, multi-</td><td></td><td></td><td></td></tr><tr><td></td><td>Gaussian， Poisson， binomial， and multinomial regression， making it versatile for different</td><td></td><td></td></tr><tr><td>types of response variables [20]. The performance ofglmnet` isfrequentlyused asabenchmarkwhen</td><td></td><td></td><td></td></tr></table></body></html>  

lassolassolassolasso lasso\` function, offering a dedicated tool for implementing Lasso and Elastic Net regularization [16]. The function comes with defined syntax, various input and output arguments, and name-value pair options, providing flexibility for users within the MATLAB environment [16].  

Beyond these established libraries, researchers also implement Lasso algorithms from scratch, often using fundamental numerical libraries like NumPy in Python [12]. Custom implementations based on methods such as coordinate descent and LARS are detailed in various works, providing insight into the algorithmic mechanics [1,25]. This approach can be valuable for educational purposes, for developing and testing novel algorithmic variants, or when specific problem constraints necessitate a tailored solution.  

The choice between using specialized Lasso solvers (like Scikit-learn or Glmnet) and more general optimization libraries involves distinct trade-offs [30]. Specialized libraries are typically highly optimized for the specific structure of the Lasso problem, often resulting in superior computational performance and providing convenient, built-in functionalities relevant to Lasso, such as efficient path computation or cross-validation support [11]. Their APIs are tailored to the statistical modeling context, enhancing ease of use for standard Lasso applications. In contrast, general optimization libraries and  

tools, which may include components for linear programming, quadratic programming, or broader mathematical utilities [30], offer greater flexibility. They can be used to formulate and solve a wider array of optimization problems, where Lasso can be represented as a specific convex optimization instance. However, employing a general solver for Lasso might require more effort in explicitly formulating the objective function and constraints and may not be as computationally efficient as a specialized solver tuned for the problem's specific structure. The decision often depends on the user's primary goal: seeking maximal convenience and performance for standard Lasso problems typically favors specialized libraries, while the need for flexibility to handle variations, combine Lasso with other optimization components, or integrate into larger optimization frameworks might lead to the use of general-purpose tools [30].  

# 9. Conclusion and Future Directions  

This survey has explored the fundamental concepts of the Lasso problem, its significance in promoting sparsity and regularization, and the diverse array of numerical algorithms developed for its solution and that of its variants. Lasso, by employing the $L _ { 1 }$ ​ penalty, serves as a powerful tool for simultaneous feature selection and regularization in linear models, crucial for preventing overfitting and enhancing generalization ability, particularly when dealing with a large number of features [5,23,27]. While effective in setting coefficients to zero for feature compression, a potential drawback of standard Lasso is the risk of incorrectly eliminating useful features, a scenario where Ridge regression might offer superior accuracy [5]. Elastic Net, combining $L _ { 1 }$ ​ and $L _ { 2 }$ ​ penalties, is often considered generally superior to Lasso, especially in settings with numerous or strongly correlated features [25].  

Beyond the basic Lasso formulation, various extensions and related penalties have emerged to address more complex sparsity structures and problem domains. For instance, the $L _ { 1 / 2 }$ ​ penalty has demonstrated advantages over ordinary $L _ { 1 }$ and Elastic Net regularization in sparse logistic regression for gene selection, achieving higher classification accuracy with fewer informative features, which is beneficial for applications like screening and diagnostics where cost control is essential [17]. Similarly, ​ L21 ​​​ minimization techniques, solvable via algorithms like ADMM, have proven efficient for sparse signal recovery in both noisy and noiseless scenarios [14]. For block-sparse signal recovery, the $\frac { l _ { 2 } } { l _ { 1 } }$ minimization method, supported by sharp sufficient conditions, shows robustness and stability [24]. Non-negative sparse signal recovery can be effectively handled by specific algorithms like RHT, validated through theoretical analysis and numerical results [10]. Furthermore, joint adaptive LASSO combined with block-sparse Bayesian approaches has shown superior performance in challenging applications such as long-range source localization under low SNR and limited snapshots, highlighting the effectiveness of adaptive sparsity priors [7]. For group sparsity problems, relaxations using penalties like the group Capped$L _ { 1 }$ have been studied, demonstrating the ability to capture underlying sparsity structures [2]. These examples underscore the importance of developing numerical algorithms tailored to specific penalty functions and problem characteristics.​  

Effective machine learning modeling with Lasso requires iterative adjustments and careful selection of the regularization parameter, $\lambda$ , to appropriately balance model complexity and generalization performance [20,22]. While various numerical methods like Coordinate Descent, Proximal Methods, and ADMM exist to solve the Lasso optimization problem and its variants, the digests highlight the need for efficient algorithms, particularly for non-convex formulations such as ​L1/2 ​ , ​ L21 ​​​ , and Capped- $L _ { 1 }$ ​ minimization [2,14,24].​  

<html><body><table><tr><td>Limitation</td><td>Corresponding Future Direction</td></tr><tr><td>Scaling to truly massive datasets</td><td>Develop more efficient& scalable algorithms (distributed, streaming).</td></tr><tr><td>Handling ill-conditioned problems</td><td>Develop robust algorithms for challenging data.</td></tr><tr><td>Optimizing non-convex formulations</td><td>Advance algorithms for non-convex penalties (e.g., Lp, Capped-L1).</td></tr><tr><td>Theoretical guarantees (non-convex, distributed)</td><td>Improve theoretical understanding under weaker/complex assumptions.</td></tr><tr><td>Manual parameter selection</td><td></td></tr></table></body></html>  

<html><body><table><tr><td></td><td>Develop adaptive/data-aware parameter selection methods.</td></tr><tr><td>Application to new domains</td><td>Explore Lasso use in emerging signal, image, bio, etc. problems.</td></tr></table></body></html>  

Despite significant progress, several limitations persist in the current research landscape. A major challenge lies in scaling algorithms to truly massive datasets, particularly in distributed or streaming settings, which demands algorithms with high efficiency and low memory footprints. Developing robust algorithms for highly ill-conditioned problems and non-convex formulations, such as those involving $L _ { p }$ ​ penalties with $p < 1$ or Capped- $L _ { 1 }$ ​ , remains an active area of research [2,14]. Furthermore, improving theoretical guarantees for these algorithms, especially under weaker assumptions or in non-convex and distributed environments, is essential [10,24].​  

These limitations pave the way for promising future research directions. Developing more efficient and scalable numerical algorithms capable of handling increasingly large datasets is paramount. Advancing the theoretical understanding of existing and new algorithms, particularly for non-convex problems and in distributed computing settings, is crucial for establishing their reliability and performance guarantees [10,24]. Exploring hybrid algorithms that combine the strengths of different optimization techniques may lead to improved performance. Developing adaptive or data-aware parameter selection methods could alleviate the need for extensive cross-validation, allowing models to dynamically adjust to data characteristics [20,22]. Finally, applying Lasso and its variants to new and emerging problem domains, including specific signal processing tasks [14], Compressive Sensing acquisition and reconstruction strategies [29], and specific biological or engineering problems like source localization or gene selection [7,17], continues to be a vital avenue for demonstrating their practical value and driving further methodological innovation.​  

# References  

[1] 机器学习：拉索回归(Lasso)算法原理及实现 https://blog.csdn.net/weixin_50804299/article/details/137924219   
[2] 组稀疏优化问题: 精确连续Capped- $L _ { 1 }$ ​ 松弛研究 https://www.actamath.com/Jwk_sxxb_cn/CN/Y2022/V65/I2/243   
[3] Lasso回归：原理、数学推导与实例 https://blog.csdn.net/m0_66813240/article/details/146565632​   
[4] 石跃勇等发表高维LASSO回归广义牛顿-拉夫逊算法研究论文 https://jgxy.cug.edu.cn/info/1171/8050.htm​   
[5] LASSO回归：模型正则化与特征选择 https://louyu.cc/?p $\vDash$ 1745   
[6] 基于非凸复合函数的稀疏信号恢复算法 http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c200666   
[7] 联合自适应LASSO与块稀疏贝叶斯的远距离辐射源直接定位 http://radarst.cnjournals.com/html/2024/3/202403004.html   
[8] Lasso回归：原理、应用与深度探索 https://blog.csdn.net/qq_51320133/article/details/137421955   
[9] 高维稀疏学习的构造性方法 http://www.amss.ac.cn/mzxsbg/202006/t20200628_5612531.html   
[10] 暨南大学温金明教授：基于ReLU的非负稀疏信号恢复硬阈值算法 https://maths.swjtu.edu.cn/info/1041/10653.htm   
[11] Lasso回归：特征选择与稀疏模型 https://blog.csdn.net/daunxx/article/details/51596877   
[12] 压缩感知技术：优缺点、算法与未来发展 https://blog.csdn.net/universsky2015/article/details/137308959   
[13] Lasso回归：原理、求解方法与总结 https://cloud.tencent.com/developer/article/2092238   
[14] L1/L2范数最小化算法在稀疏信号恢复中的应用与收敛性保证 http://math.tju.edu.cn/info/1059/7112.htm   
[15] 基于正交多项式的正则化加权最小二乘法 https://maths.whu.edu.cn/info/1115/10473.htm   
[16] Lasso and Elastic Net Regularization for Linear Mo https://www.mathworks.com/help/stats/lasso.html   
[17] L1/2 Penalized Sparse Logistic Regression for Gene http://dx.doi.org/10.1186/1471-2105-14-198   
[18] Linear Models: Regression and Classification https://scikit-learn.org/1.0/modules/linear_model.html​   
[19] 喻文健的论著选集 https://numbda.cs.tsinghua.edu.cn/paper.html​   
[20] Lasso回归：原理推导、偏差方差与实战 https://mp.weixin.qq.com/s? _biz=MzAwODIzOTExNw $\scriptstyle 1 = =$ &mid=2247484746&idx $\mathop { : = }$ 1&sn=005497e371bf12271f84ef03518be0a1&chksm $\mid =$ 9b70a428ac072d   
f0f93ac6ee33c21cb30dc9e2fbd1f61174ef5ca43f4344fb159a81601b3f&scene=27   
[21] Lasso回归：原理与实例 https://blog.csdn.net/weixin_47151388/article/details/138212528   
[22] 机器学习初学者教程：L1和L2正则化详解 https://cloud.tencent.com/developer/article/1118616?policyId=1004   
[23] 机器学习正则化：防止过拟合与提升泛化能力 https://zhidao.baidu.com/question/1780127595415051020.html   
[24] Sharp Condition for Block Sparse Signal Recovery v http://dx.doi.org/10.1049/iet-spr.2018.5037   
[25] Lasso回归算法详解与实现 https://blog.csdn.net/dalangtaosha999/article/details/129711052   
[26] L1和L2正则化详解：多角度分析与比较 https://baijiahao.baidu.com/s?id $\ c =$ 1621054167310242353&wfr=spider&for=pc   
[27] LASSO值得学吗？原理、应用、优缺点全解析 https://qianfanmarket.baidu.com/article/detail/272075   
[28] 统计学学术速递[7.8]论文标题精简 https://cloud.tencent.com/developer/article/1852707   
[29] 压缩感知：信号处理领域的新利器 https://mp.weixin.qq.com/s? _biz $: =$ MzIxMzI4ODI1MA $\scriptstyle 1 = =$ &mid $=$ 2247484546&idx $\mathrel { \mathop : }$ 1&sn $\ c =$ c19d0a21004ae2d7b0623f964fdd5b3e&chksm $\mid =$ 97b850d7a0cfd9c18   
bfe94fa3a76b4e12cddbf388c25fd081bbc228d87934ed8b18d1edce08a&scene=27   
[30] Open-Source Solvers for Optimization & Problem Sol https://github.com/topics/solver  