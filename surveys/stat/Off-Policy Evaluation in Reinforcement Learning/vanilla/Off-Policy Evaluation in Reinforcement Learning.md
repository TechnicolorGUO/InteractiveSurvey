# Off-Policy Evaluation in Reinforcement Learning

## Introduction

Off-Policy Evaluation (OPE) is a critical component of reinforcement learning (RL), enabling the assessment of a target policy's performance using data collected by a different behavior policy. This capability is particularly valuable in real-world applications where direct experimentation with the target policy may be costly, risky, or unethical. OPE methods aim to estimate the expected return of a policy without requiring interaction with the environment under that policy. In this survey, we provide an overview of the key concepts, methodologies, and challenges associated with OPE.

## Background and Motivation

Reinforcement learning involves an agent interacting with an environment to maximize cumulative rewards. Policies define how actions are selected given states, and evaluating their performance is essential for improving decision-making strategies. However, when the policy to be evaluated differs from the one used to collect data, traditional on-policy evaluation techniques become infeasible. This necessitates the development of off-policy evaluation methods.

The importance of OPE spans various domains, including healthcare, robotics, recommendation systems, and autonomous driving. For instance, in healthcare, OPE allows researchers to evaluate treatment policies using observational data without exposing patients to experimental treatments.

## Key Concepts

### Importance Sampling

Importance sampling is a foundational technique in OPE. It adjusts the distribution of observed trajectories to match that of the target policy. The estimator can be expressed as:

$$
\hat{V}_{IS}(\pi_e) = \frac{1}{N} \sum_{i=1}^N \rho_i G_i,
$$

where $G_i$ is the discounted return of trajectory $i$, and $\rho_i = \prod_t \frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)}$ is the importance weight reflecting the ratio of probabilities under the target ($\pi_e$) and behavior ($\pi_b$) policies.

While straightforward, importance sampling suffers from high variance, especially in long-horizon problems.

### Doubly Robust Estimators

To address the variance issue, doubly robust (DR) estimators combine importance sampling with model-based predictions. A DR estimator can be written as:

$$
\hat{V}_{DR}(\pi_e) = \frac{1}{N} \sum_{i=1}^N \left( \rho_i (G_i - Q(s_0, a_0)) + Q(s_0, a_0) \right),
$$

where $Q(s, a)$ is an estimated value function. DR estimators maintain consistency if either the importance weights or the value function estimates are accurate, offering improved robustness.

### Model-Based Approaches

Model-based methods involve learning a dynamics model of the environment and using it to simulate trajectories under the target policy. These approaches avoid reliance on importance weights but require accurate modeling of the environment. A common challenge is model bias, which can propagate errors in simulated returns.

### Fitted Q-Evaluation

Fitted Q-evaluation (FQE) leverages function approximation to estimate the value function of the target policy. By iteratively updating the Q-function using observed transitions, FQE provides a flexible and scalable solution for OPE. However, its performance depends on the quality of the learned Q-function.

## Advanced Techniques

### Variance Reduction

Several techniques have been proposed to reduce the variance of importance sampling estimators. Per-decision importance sampling and stationary importance sampling modify the weighting scheme to account for temporal dependencies and stabilize estimates.

### Causal Inference Perspectives

Causal inference offers a complementary perspective on OPE, framing it as a problem of estimating counterfactual outcomes. Methods such as inverse propensity scoring and outcome regression align closely with importance sampling and value estimation in RL.

### Offline Reinforcement Learning

Offline RL extends OPE by incorporating policy optimization within the constraints of logged data. Techniques like conservative Q-learning and pessimistic policy optimization ensure robustness in the absence of additional exploration.

| Technique | Strengths | Weaknesses |
|-----------|-----------|------------|
| Importance Sampling | Conceptually simple | High variance |
| Doubly Robust | Reduced variance | Requires accurate models |
| Model-Based | Avoids importance weights | Sensitive to model bias |
| FQE | Scalable | Dependent on function approximation |

## Challenges and Open Problems

Despite significant progress, OPE faces several challenges:

- **High-Dimensional State Spaces**: Handling large state spaces exacerbates variance issues and model inaccuracies.
- **Distributional Shifts**: Behavior and target policies often differ significantly, complicating the alignment of data distributions.
- **Confounding Variables**: Unobserved confounders in observational data can introduce biases in estimates.
- **Generalization**: Ensuring reliable performance across diverse environments remains an open research question.

![](placeholder_for_distribution_shift_diagram)

## Conclusion

Off-Policy Evaluation is a vibrant area of research with profound implications for practical RL applications. While classical methods like importance sampling and doubly robust estimators form the backbone of OPE, recent advances in variance reduction, causal inference, and offline RL continue to push the boundaries of what is possible. Addressing the challenges outlined above will be crucial for realizing the full potential of OPE in real-world scenarios.

Future work should focus on developing more robust and efficient algorithms, integrating insights from causal reasoning, and exploring hybrid approaches that combine the strengths of existing techniques.
