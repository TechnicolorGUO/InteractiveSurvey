# 1 Introduction

The analysis of longitudinal and panel data has become increasingly important in fields such as economics, epidemiology, sociology, and political science. These datasets allow researchers to study dynamic processes over time, capturing both within-unit changes and between-unit differences. However, establishing causal relationships in such settings is challenging due to the presence of confounding factors, temporal dependencies, and heterogeneity across units. This survey provides a comprehensive overview of causal modeling techniques tailored for longitudinal and panel data.

## 1.1 Motivation for Causal Modeling in Longitudinal and Panel Data

Longitudinal and panel data are characterized by repeated observations of the same units over time. Such data structures offer unique opportunities to disentangle causation from correlation. For instance, they enable the examination of how interventions or exposures affect outcomes over time, accounting for individual-specific characteristics that may otherwise bias estimates. Despite these advantages, causal inference in longitudinal and panel data poses several challenges:

- **Time-varying confounders**: Variables that change over time can simultaneously influence both the treatment assignment and the outcome, leading to biased estimates if not properly addressed.
- **Temporal dependencies**: Observations within the same unit are often correlated, violating assumptions of independence required by many standard statistical methods.
- **Heterogeneity**: Units may differ systematically in ways that affect their responses to treatments, necessitating models that account for such variability.

To address these issues, specialized causal modeling techniques have been developed. These methods aim to estimate causal effects while accounting for the complexities inherent in longitudinal and panel data.

## 1.2 Objectives of the Survey

This survey aims to achieve the following objectives:

1. **Provide an overview of fundamental concepts in causal inference**: We introduce key frameworks, such as the potential outcomes framework and directed acyclic graphs (DAGs), which form the foundation for understanding causal relationships.
2. **Review causal modeling techniques for longitudinal data**: We discuss methods like marginal structural models (MSMs), g-estimation, and structural nested models (SNMs) that are specifically designed to handle time-dependent treatments and confounders.
3. **Examine causal modeling approaches for panel data**: We explore fixed effects and random effects models, difference-in-differences (DiD), and synthetic control methods, highlighting their strengths and limitations.
4. **Discuss advanced topics**: We delve into machine learning-based causal inference, Bayesian approaches, and strategies for addressing time-varying confounders and mediators.
5. **Synthesize insights and identify open research questions**: By comparing the various methods discussed, we highlight practical considerations and outline promising directions for future work.

## 1.3 Structure of the Paper

The remainder of this survey is organized as follows:

- **Section 2** provides background information on causal inference, including the potential outcomes framework, DAGs, and the nature of longitudinal and panel data.
- **Section 3** focuses on causal models for longitudinal data, covering MSMs, g-estimation, and SNMs.
- **Section 4** examines causal models for panel data, discussing fixed/random effects models, DiD, and synthetic control methods.
- **Section 5** explores advanced topics, such as machine learning approaches, time-varying confounders, and Bayesian causal inference.
- **Section 6** offers a comparative discussion of the methods reviewed, along with practical considerations and open research questions.
- Finally, **Section 7** summarizes the key findings and implications for future research.

Throughout the survey, we emphasize the importance of selecting appropriate methods based on the specific features of the data and the research question at hand.

# 2 Background

To understand causal modeling for longitudinal and panel data, it is essential to establish a foundational understanding of causal inference and the unique characteristics of these types of data. This section provides an overview of the key concepts in causal inference and highlights the specific features of longitudinal and panel datasets that influence causal analysis.

## 2.1 Fundamentals of Causal Inference

Causal inference aims to estimate the effect of interventions or treatments on outcomes while accounting for confounding factors. Below, we discuss three fundamental components: the potential outcomes framework, directed acyclic graphs (DAGs), and confounding and bias.

### 2.1.1 Potential Outcomes Framework

The potential outcomes framework, also known as the Rubin causal model, formalizes causality by defining counterfactuals. For each unit $i$, there exist two potential outcomes: $Y_i(1)$ under treatment and $Y_i(0)$ under control. The causal effect for unit $i$ is defined as:

$$
\text{Causal Effect} = Y_i(1) - Y_i(0)
$$

In practice, only one of the potential outcomes is observed, leading to the "fundamental problem of causal inference." To address this, assumptions such as ignorability ($Y_i(1), Y_i(0) \perp T_i | X_i$) are invoked, where $T_i$ denotes treatment assignment and $X_i$ represents observed covariates.

### 2.1.2 Directed Acyclic Graphs (DAGs)

Directed acyclic graphs (DAGs) provide a visual and mathematical tool for representing causal relationships among variables. A DAG consists of nodes (variables) and directed edges (causal pathways). By applying rules such as d-separation, researchers can identify which variables need to be conditioned upon to eliminate confounding. For example, in a simple DAG with $X \rightarrow Y \leftarrow Z$, $Z$ is a collider, and conditioning on it induces spurious associations unless carefully managed.

![](placeholder_for_dag_image)

### 2.1.3 Confounding and Bias

Confounding occurs when a variable influences both the treatment and the outcome, distorting the estimated causal effect. Bias arises if confounding is not properly addressed. Common strategies to mitigate confounding include stratification, matching, and inverse probability weighting. However, unmeasured confounding remains a persistent challenge, requiring sensitivity analyses to assess robustness.

## 2.2 Characteristics of Longitudinal and Panel Data

Longitudinal and panel data possess distinct features that complicate causal inference. These include temporal dependencies, heterogeneity across units, and missing data challenges.

### 2.2.1 Temporal Dependencies

Temporal dependencies arise because observations within the same unit over time are often correlated. This autocorrelation violates the assumption of independence in standard regression models. Methods such as generalized estimating equations (GEEs) or mixed-effects models account for these dependencies by incorporating lagged effects or random intercepts/slopes.

### 2.2.2 Heterogeneity Across Units

Units in longitudinal and panel data may exhibit substantial heterogeneity in their responses to treatments. Ignoring this heterogeneity can lead to biased estimates. Fixed effects models capture unit-specific variation by including dummy variables for each unit, while random effects models assume unit-specific effects follow a distribution.

### 2.2.3 Missing Data Challenges

Missing data are prevalent in longitudinal studies due to attrition or non-response. Missingness mechanisms—missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR)—determine appropriate imputation techniques. Multiple imputation and full information maximum likelihood (FIML) are commonly used to handle missing data while preserving statistical power.

| Missingness Mechanism | Description |
|-----------------------|-------------|
| MCAR                 | Missingness is unrelated to any variables. |
| MAR                  | Missingness depends on observed variables. |
| MNAR                 | Missingness depends on unobserved values. |

# 3 Causal Models for Longitudinal Data

Longitudinal data, characterized by repeated observations over time for the same units, presents unique challenges and opportunities for causal inference. This section reviews three prominent classes of causal models tailored to longitudinal settings: Marginal Structural Models (MSMs), G-estimation for dynamic treatment regimes, and Structural Nested Models (SNMs). Each approach addresses specific complexities inherent in longitudinal data, such as time-varying confounding and sequential decision-making.

## 3.1 Marginal Structural Models (MSMs)

Marginal Structural Models (MSMs) are a class of causal models designed to estimate population-level effects in the presence of time-varying confounders influenced by prior treatments. MSMs achieve this by weighting observations using inverse probability weights (IPWs), which balance covariate distributions across treatment groups at each time point.

### 3.1.1 Estimation Techniques

The estimation of MSMs involves two primary steps: (1) modeling the propensity scores for treatment assignment and (2) fitting a weighted regression model. The propensity score $P(A_t = 1 | L_t, A_{t-1})$ represents the probability of receiving treatment $A_t$ given the history of covariates $L_t$ and past treatments $A_{t-1}$. IPWs are then calculated as:

$$
W_i = \prod_{t=1}^T \frac{P(A_t = 1 | L_t, A_{t-1})}{P(A_t = 0 | L_t, A_{t-1})}
$$

These weights are applied to a marginal structural model of the form:

$$
E[Y | A] = g(\beta_0 + \beta_1 A)
$$

where $Y$ is the outcome, $A$ is the treatment, and $g(\cdot)$ is a link function (e.g., identity or logit).

| Estimation Step | Description |
|-----------------|-------------|
| Propensity Score Modeling | Estimate probabilities of treatment assignment. |
| Weight Calculation | Compute inverse probability weights. |
| Weighted Regression | Fit the MSM using weighted least squares or generalized estimating equations. |

### 3.1.2 Advantages and Limitations

Advantages of MSMs include their ability to handle complex time-varying confounding structures and provide interpretable estimates of average treatment effects. However, limitations arise from reliance on correctly specified propensity score models and potential instability due to extreme weights. Sensitivity analyses are often recommended to assess robustness.

## 3.2 G-estimation for Dynamic Treatment Regimes

G-estimation extends traditional regression methods to account for sequential causal effects in longitudinal settings. It is particularly suited for evaluating dynamic treatment regimes, where treatment decisions depend on evolving patient characteristics.

### 3.2.1 Sequential Causal Effects

Sequential causal effects refer to the impact of treatments administered at multiple time points. G-estimation leverages the g-formula to estimate counterfactual outcomes under hypothetical treatment strategies. For example, consider a sequence of treatments $A_1, A_2, \dots, A_T$. The g-formula expresses the expected outcome as:

$$
E[Y] = \int E[Y | A_1, A_2, \dots, A_T, L_1, L_2, \dots, L_T] f(L_1, L_2, \dots, L_T | A_1, A_2, \dots, A_T) dL
$$

This decomposition allows for the estimation of causal effects while adjusting for time-dependent confounders.

### 3.2.2 Application to Panel Data

In panel data settings, G-estimation can be adapted to incorporate unit-specific heterogeneity. By modeling individual-level trajectories, researchers can better account for unobserved confounders that vary across units. ![](placeholder_for_g_estimation_diagram)

## 3.3 Structural Nested Models (SNMs)

Structural Nested Models (SNMs) focus on nested counterfactuals, allowing for the estimation of causal effects in settings with complex temporal dependencies.

### 3.3.1 Nested Counterfactuals

Nested counterfactuals extend standard counterfactual reasoning to account for interactions between past and future treatments. For instance, the effect of treatment $A_t$ may depend on the history of treatments $A_{t-1}, A_{t-2}, \dots$. SNMs express these relationships through recursive structural equations:

$$
Y_t = g(Y_{t-1}, A_t, L_t, \epsilon_t)
$$

where $\epsilon_t$ captures residual variation.

### 3.3.2 Implementation Challenges

Implementing SNMs requires careful specification of the structural equations and assumptions about the functional form of $g(\cdot)$. Additionally, estimation often relies on specialized algorithms, such as iterative conditional expectations, which can be computationally intensive. Despite these challenges, SNMs offer a flexible framework for addressing intricate causal questions in longitudinal studies.

# 4 Causal Models for Panel Data

Panel data, which combines cross-sectional and time-series information, offers a rich setting for causal inference. This section reviews key methods tailored to panel data, including fixed effects and random effects models, difference-in-differences (DiD), and synthetic control methods. Each approach addresses specific challenges inherent in panel data, such as unobserved heterogeneity and temporal dependencies.

## 4.1 Fixed Effects and Random Effects Models

Fixed effects (FE) and random effects (RE) models are foundational tools for analyzing panel data. These models account for unobserved heterogeneity that may bias causal estimates if ignored.

### 4.1.1 Within-Unit Variation

The fixed effects model exploits within-unit variation by differencing out time-invariant characteristics. Mathematically, the model can be expressed as:
$$
y_{it} = \alpha_i + X_{it}\beta + \epsilon_{it},
$$
where $y_{it}$ is the outcome for unit $i$ at time $t$, $\alpha_i$ represents the unit-specific fixed effect, $X_{it}$ denotes observed covariates, and $\epsilon_{it}$ is the idiosyncratic error term. By focusing on changes over time within each unit, FE models effectively control for all time-invariant confounders captured in $\alpha_i$.

### 4.1.2 Time-Invariant Confounders

Random effects models, in contrast, treat $\alpha_i$ as a random variable drawn from a distribution. The RE framework assumes that $\alpha_i$ is uncorrelated with the covariates $X_{it}$, allowing for more efficient estimation compared to FE when this assumption holds. However, if $\alpha_i$ is correlated with $X_{it}$, the RE estimator will be biased, making FE the preferred choice in such cases.

| Model | Key Assumptions | Strengths | Limitations |
|-------|-----------------|-----------|-------------|
| Fixed Effects | $\alpha_i$ correlates with $X_{it}$ | Controls for all time-invariant confounders | Loses efficiency due to reliance on within-unit variation |
| Random Effects | $\alpha_i$ uncorrelated with $X_{it}$ | More efficient, uses both within- and between-unit variation | Sensitive to violations of the uncorrelation assumption |

## 4.2 Difference-in-Differences (DiD) Approach

The DiD method compares changes over time between a treatment group and a control group to estimate causal effects. This approach relies on the parallel trends assumption, which posits that, absent treatment, the average outcomes of the treated and control groups would have followed the same trajectory.

### 4.2.1 Parallel Trends Assumption

Formally, the parallel trends assumption implies:
$$
E[Y^0_{it} | D_i=1] - E[Y^0_{it} | D_i=0] = \text{constant for all } t,
$$
where $Y^0_{it}$ is the counterfactual outcome for unit $i$ at time $t$ under no treatment, and $D_i$ indicates treatment status. Violations of this assumption can lead to biased estimates, necessitating robustness checks or alternative methods.

### 4.2.2 Extensions to Nonlinear Models

While traditional DiD is often implemented using linear regression, extensions to nonlinear settings, such as logistic or probit models, allow for more flexible functional forms. For example, in a binary outcome context, the average treatment effect on the treated (ATT) can be estimated as:
$$
ATT = P(Y_{it}=1 | D_i=1) - P(Y_{it}=1 | D_i=0),
$$
where probabilities are derived from the nonlinear model. These extensions broaden the applicability of DiD but require careful interpretation of results.

## 4.3 Synthetic Control Methods

Synthetic control methods construct counterfactual outcomes for treated units by weighting untreated units to approximate their pre-treatment behavior. This approach is particularly useful when a single treated unit is analyzed relative to multiple controls.

### 4.3.1 Construction of Counterfactuals

The synthetic control is formed as a weighted average of control units' outcomes, where weights are chosen to minimize discrepancies in pre-treatment covariates and trends. Let $W = (w_1, w_2, ..., w_J)$ denote the vector of weights for $J$ control units. The synthetic control outcome is then given by:
$$
Y^0_{it} = \sum_{j=1}^J w_j Y_{jt},
$$
subject to constraints ensuring non-negativity and normalization of weights ($\sum_{j=1}^J w_j = 1$).

![](placeholder_for_synthetic_control_diagram)

### 4.3.2 Evaluating Model Fit

Assessing the quality of the synthetic control involves comparing pre-treatment outcomes of the treated unit and its synthetic counterpart. A good fit ensures that the constructed counterfactual closely tracks the treated unit's trajectory prior to intervention, lending credibility to post-treatment causal inferences. Common metrics include mean squared prediction error (MSPE) and visual diagnostics of pre-treatment alignment.

# 5 Advanced Topics in Causal Modeling

In this section, we delve into advanced methodologies for causal inference that address complex challenges inherent in longitudinal and panel data. These approaches extend beyond traditional statistical models to incorporate machine learning techniques, account for time-varying confounders and mediators, and leverage Bayesian frameworks for more nuanced causal analyses.

## 5.1 Machine Learning Approaches

Machine learning (ML) methods have gained prominence in causal inference due to their ability to handle high-dimensional data and complex interactions without strong parametric assumptions. Below, we discuss two prominent ML-based approaches: causal forests and double/debiased machine learning.

### 5.1.1 Causal Forests

Causal forests are an extension of random forests tailored for estimating heterogeneous treatment effects. They partition the covariate space into regions where treatment effects vary, providing a flexible approach to modeling individual-level heterogeneity. The key idea is to estimate conditional average treatment effects (CATEs) using recursive partitioning. Formally, the CATE is defined as:
$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x],
$$
where $Y(1)$ and $Y(0)$ represent potential outcomes under treatment and control, respectively, and $X$ denotes the set of covariates. Causal forests achieve this by modifying the splitting criterion of standard decision trees to prioritize splits that maximize differences in treatment effects across child nodes.

![](placeholder_for_causal_forest_diagram)

### 5.1.2 Double/Debiased Machine Learning

Double/debiased machine learning (DML) addresses the challenge of bias introduced by regularization in high-dimensional settings. DML separates the estimation of nuisance parameters (e.g., propensity scores or outcome regressions) from the estimation of causal effects, ensuring robustness even when nuisance models are misspecified. This method relies on orthogonal score equations, which reduce sensitivity to first-stage estimation errors. For example, in the context of linear models, the DML estimator can be expressed as:
$$
\hat{\tau} = \frac{1}{n} \sum_{i=1}^n \left[ \frac{T_i(Y_i - \hat{m}_0(X_i))}{\hat{e}(X_i)} + \hat{m}_1(X_i) - \hat{m}_0(X_i) \right],
$$
where $T_i$ is the treatment indicator, $Y_i$ is the outcome, $X_i$ are covariates, and $\hat{m}_0$, $\hat{m}_1$, and $\hat{e}$ are estimated nuisance functions.

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| Causal Forests | Flexible, handles heterogeneity well | Computationally intensive |
| DML | Robust to model misspecification | Requires careful tuning of nuisance estimators |

## 5.2 Time-Varying Confounders and Mediators

Longitudinal studies often involve time-varying confounders and mediators, complicating causal inference. We explore two techniques designed to address these complexities: the G-computation algorithm and sensitivity analysis.

### 5.2.1 G-computation Algorithm

The G-computation algorithm estimates causal effects in the presence of time-varying confounders affected by prior treatments. It involves simulating potential outcomes under different treatment regimes by iteratively predicting outcomes conditional on observed histories. Mathematically, the G-computation formula for a sequence of treatments $A_0, A_1, ..., A_K$ is given by:
$$
\mathbb{E}[Y(a_0, a_1, ..., a_K)] = \int ... \int \prod_{k=0}^K f(Y_k | A_k = a_k, L_k) f(L_k | A_{k-1}, L_{k-1}) dL_0...dL_K,
$$
where $L_k$ represents time-varying covariates at time $k$. This approach requires correct specification of all relevant models but offers a principled way to handle dynamic treatment regimes.

### 5.2.2 Sensitivity Analysis

Sensitivity analysis evaluates how robust causal estimates are to unmeasured confounding. In longitudinal settings, this involves quantifying the impact of hypothetical unobserved variables on estimated effects. One common approach is to define a sensitivity parameter $\delta$, representing the strength of association between unmeasured confounders and both treatment and outcome. By varying $\delta$, researchers can assess whether conclusions remain consistent under plausible deviations from assumptions.

## 5.3 Bayesian Causal Inference

Bayesian methods offer a natural framework for incorporating prior knowledge and uncertainty into causal inference. We focus on hierarchical Bayesian models and the role of prior specification in causal analysis.

### 5.3.1 Hierarchical Bayesian Models

Hierarchical Bayesian models (HBMs) are particularly suited for panel data with grouped structures, allowing for partial pooling of information across units. HBMs specify priors over unit-specific parameters, enabling shrinkage toward population-level estimates. For instance, consider a linear mixed-effects model:
$$
Y_{it} = \beta_0 + \beta_1 T_{it} + u_i + \epsilon_{it},
$$
where $u_i \sim N(0, \sigma_u^2)$ captures unit-specific random effects, and $\epsilon_{it} \sim N(0, \sigma_e^2)$ represents idiosyncratic error. Bayesian inference proceeds via Markov Chain Monte Carlo (MCMC), yielding posterior distributions for all parameters.

### 5.3.2 Prior Specification and Interpretation

Careful prior specification is critical in Bayesian causal inference. Weakly informative priors can stabilize estimates while avoiding undue influence, whereas structured priors may encode domain knowledge about causal relationships. However, priors must be interpreted cautiously, as they can introduce bias if misaligned with reality. Sensitivity analyses comparing results under different priors help ensure robustness.



# 7 Conclusion

In this concluding section, we synthesize the key findings from our survey on causal modeling for longitudinal and panel data, while also highlighting implications for future research.

## 7.1 Summary of Key Findings

This survey has explored the theoretical underpinnings and practical applications of causal inference methods tailored to longitudinal and panel data. Below is a summary of the main insights:

1. **Causal Inference Fundamentals**: The potential outcomes framework and directed acyclic graphs (DAGs) provide essential tools for understanding causal relationships in complex datasets. Confounding and bias remain central challenges, requiring careful identification strategies.

2. **Characteristics of Longitudinal and Panel Data**: Temporal dependencies, heterogeneity across units, and missing data complicate causal analyses. These features necessitate specialized models that account for within-unit variation and time-invariant confounders.

3. **Causal Models for Longitudinal Data**: Marginal structural models (MSMs), g-estimation, and structural nested models (SNMs) are powerful techniques for addressing dynamic treatment regimes and sequential causal effects. For example, MSMs estimate causal effects by weighting observations inversely proportional to their probability of receiving treatment: $\hat{\beta} = \sum_{i=1}^N w_i y_i / \sum_{i=1}^N w_i$, where $w_i$ represents stabilized inverse probability weights.

4. **Causal Models for Panel Data**: Fixed effects, random effects, difference-in-differences (DiD), and synthetic control methods address specific challenges in panel settings. DiD relies on the parallel trends assumption, which can be tested using pre-treatment covariates. Synthetic control methods construct counterfactuals through weighted averages of control units, as shown in Table | Column 1 | Column 2 |.

5. **Advanced Topics**: Machine learning approaches, such as causal forests and double/debiased machine learning, enhance flexibility in high-dimensional settings. Bayesian methods offer probabilistic interpretations of uncertainty, with hierarchical models accommodating varying levels of complexity.

6. **Time-Varying Confounders and Mediators**: Addressing time-varying confounders requires sophisticated algorithms like g-computation, which iteratively predicts outcomes under hypothetical interventions. Sensitivity analysis ensures robustness against unmeasured confounding.

## 7.2 Implications for Future Work

While significant progress has been made in causal modeling for longitudinal and panel data, several open questions warrant further exploration:

1. **Integration of Machine Learning and Causal Inference**: Developing hybrid methods that leverage the strengths of both paradigms could improve estimation accuracy and interpretability. For instance, combining causal forests with traditional econometric techniques may yield more reliable estimates.

2. **Handling Missing Data**: Advanced imputation strategies are needed to address missingness patterns common in longitudinal studies. Multiple imputation and full information maximum likelihood (FIML) approaches deserve further investigation.

3. **Scalability to Large Datasets**: As datasets grow in size and complexity, scalable algorithms for causal inference become increasingly important. Distributed computing frameworks and approximate inference methods may facilitate analyses of big data.

4. **Causal Discovery Algorithms**: Automating the process of identifying causal structures from observational data remains an active area of research. Advances in DAG learning and constraint-based methods hold promise for uncovering hidden causal relationships.

5. **Ethical Considerations**: Ensuring fairness, transparency, and accountability in causal modeling is critical, particularly in sensitive domains such as healthcare and policy evaluation. Guidelines for responsible use of causal inference methods should be developed and disseminated.

In conclusion, this survey provides a comprehensive overview of causal modeling techniques for longitudinal and panel data, emphasizing both foundational principles and cutting-edge developments. By addressing the outlined research gaps, future work can further enhance our ability to draw valid causal conclusions from complex datasets.

