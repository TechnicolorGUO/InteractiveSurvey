# 1 Introduction
Reinforcement learning (RL) has emerged as a powerful paradigm for solving sequential decision-making problems, with applications ranging from robotics to healthcare. A critical challenge in RL is the ability to evaluate and improve policies without requiring interaction with the environment during the evaluation phase. This process, known as off-policy evaluation (OPE), enables practitioners to assess the performance of a target policy using data collected under a different behavior policy. In this survey, we provide an in-depth exploration of OPE methods, their theoretical foundations, practical considerations, and recent advancements.

## 1.1 Problem Definition
The problem of off-policy evaluation can be formally defined as follows: given a dataset $D = \{(s_t, a_t, r_t, s_{t+1})\}_{t=1}^T$ collected by executing a behavior policy $\pi_b$, estimate the expected return of a target policy $\pi_e$. Mathematically, this involves estimating:
$$
V^{\pi_e}(s_0) = \mathbb{E}_{\pi_e}\left[\sum_{t=0}^T \gamma^t r_t \mid s_0\right],
$$
where $\gamma \in [0, 1]$ is the discount factor, $r_t$ is the reward at time $t$, and $s_0$ is the initial state. The key challenge lies in accurately estimating this value without directly interacting with the environment under $\pi_e$, which is particularly important in settings where online experimentation is costly or unethical.

## 1.2 Importance of Off-Policy Evaluation
Off-policy evaluation plays a pivotal role in real-world applications of reinforcement learning. For instance, in healthcare, it allows researchers to evaluate treatment policies using observational data without exposing patients to experimental regimens. Similarly, in robotics, OPE enables the assessment of new control strategies using historical interaction data, reducing the need for potentially destructive physical trials. Furthermore, OPE facilitates safe and efficient policy optimization in high-stakes domains, such as finance and autonomous driving, where mistakes can have severe consequences. By decoupling evaluation from data collection, OPE bridges the gap between simulation-based training and real-world deployment.

## 1.3 Objectives and Scope
This survey aims to provide a comprehensive overview of off-policy evaluation techniques, categorizing them into importance sampling-based methods, model-based approaches, and doubly robust estimators. We will delve into their theoretical underpinnings, analyze their strengths and limitations, and discuss recent advances that leverage deep learning and domain-specific applications. Additionally, we highlight open challenges in the field, such as handling high-dimensional state spaces and partial observability, and outline promising directions for future research. While our focus is primarily on tabular and function approximation settings, we also touch upon specialized techniques tailored for continuous control and large-scale systems.

# 2 Background

To provide a comprehensive understanding of off-policy evaluation in reinforcement learning (RL), it is essential to establish the foundational concepts and terminologies that underpin this field. This section delves into the core principles of RL, differentiating between on-policy and off-policy methods while highlighting the unique challenges associated with off-policy learning.

## 2.1 Reinforcement Learning Fundamentals

Reinforcement learning revolves around an agent interacting with an environment to maximize cumulative rewards over time. The interaction process can be modeled as a Markov Decision Process (MDP), which forms the basis for most RL algorithms. Below, we outline the key components of RL and their mathematical formulation.

### 2.1.1 Markov Decision Processes

An MDP is defined by a tuple $(S, A, P, R, \gamma)$, where:
- $S$ represents the set of states,
- $A$ denotes the set of actions,
- $P(s'|s,a)$ defines the transition probability from state $s$ to $s'$ given action $a$,
- $R(s,a,s')$ specifies the expected reward for transitioning from $s$ to $s'$ via $a$, and
- $\gamma \in [0, 1]$ is the discount factor governing the importance of future rewards.

The goal of the agent is to learn a policy $\pi: S \to A$ that maximizes the expected discounted return $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ starting from time step $t$. ![](placeholder_for_mdp_diagram)

### 2.1.2 Policies and Value Functions

A policy $\pi(a|s)$ determines the probability of selecting an action $a$ in state $s$. Two primary value functions are used to evaluate policies:
- The **state-value function** $V^\pi(s) = \mathbb{E}[G_t | S_t = s]$, representing the expected return starting from state $s$ under policy $\pi$,
- The **action-value function** $Q^\pi(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$, indicating the expected return after taking action $a$ in state $s$ and subsequently following $\pi$.

These functions satisfy the Bellman equations, which decompose the value of a state or action into immediate rewards and discounted future values:
$$
V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')],
$$
$$
Q^\pi(s,a) = \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s',a')].
$$

## 2.2 On-Policy vs Off-Policy Methods

Reinforcement learning methods can broadly be classified into on-policy and off-policy approaches based on how they interact with the data-generating policy.

### 2.2.1 Characteristics of On-Policy Methods

On-policy methods estimate the value of a policy using data generated by the same policy. For example, Monte Carlo methods and Temporal Difference (TD) learning with $\epsilon$-greedy exploration fall under this category. These methods are straightforward to implement but require continuous interaction with the environment, making them less suitable for scenarios where data collection is expensive or impractical.

Key limitations include:
- High sample inefficiency due to reliance on current policy data,
- Difficulty in reusing historical data collected under different policies.

### 2.2.2 Challenges in Off-Policy Learning

Off-policy methods aim to evaluate or improve a target policy using data generated by a behavior policy, potentially distinct from the target. While this offers greater flexibility and efficiency, several challenges arise:

1. **Distribution Shift**: The discrepancy between the behavior and target policies leads to mismatched state-action distributions, complicating accurate estimation.
2. **High Variance**: Importance sampling techniques, commonly used in off-policy evaluation, often suffer from high variance when the behavior and target policies differ significantly.
3. **Bias-Variance Trade-offs**: Balancing bias and variance is critical, as overly aggressive corrections for distribution shifts may introduce significant errors.

Addressing these challenges requires sophisticated techniques, which will be explored in subsequent sections. | Key Differences | On-Policy | Off-Policy |
|------------------|------------|-------------|
| Data Source      | Same Policy | Different Policies |
| Sample Efficiency| Low         | High         |
| Complexity       | Moderate    | High         |

# 3 Literature Review

Off-policy evaluation (OPE) in reinforcement learning (RL) is a critical area of study that enables the assessment of a target policy using data generated by a different behavior policy. This section provides an extensive review of the literature on OPE, focusing on key methodologies and recent advances.

## 3.1 Importance Sampling Techniques
Importance sampling (IS) techniques are foundational to off-policy evaluation due to their ability to reweight trajectories sampled under one policy to estimate the performance of another. Below, we delve into three prominent variants of IS: basic importance sampling, per-decision importance sampling, and stationary importance sampling.

### 3.1.1 Basic Importance Sampling
Basic importance sampling (BIS) computes the ratio of probabilities between the target policy $\pi$ and the behavior policy $\mu$. For a trajectory $\tau = (s_0, a_0, r_0, \dots, s_T)$, the BIS weight is defined as:
$$
w(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}.
$$
While simple and theoretically grounded, BIS suffers from high variance when the policies differ significantly. This limitation motivates more advanced IS techniques.

### 3.1.2 Per-Decision Importance Sampling
Per-decision importance sampling (PDIS) addresses the variance issue of BIS by normalizing the weights at each time step. Specifically, PDIS calculates the cumulative reward up to time $T$ as:
$$
Q(s_0, a_0) = \sum_{t=0}^{T-1} \gamma^t r_t \prod_{k=0}^{t-1} \frac{\pi(a_k|s_k)}{\mu(a_k|s_k)}.
$$
PDIS reduces variance compared to BIS but still struggles with long-horizon problems where the product of ratios grows exponentially.

### 3.1.3 Stationary Importance Sampling
Stationary importance sampling (SIS) further mitigates variance by leveraging the stationary distribution of states induced by the behavior policy. SIS estimates the value function as:
$$
V^{\pi}(s) = \mathbb{E}_{\tau \sim \mu}[w(\tau) G(\tau)],
$$
where $G(\tau)$ represents the discounted return. By incorporating the stationary distribution, SIS achieves better stability than both BIS and PDIS.

## 3.2 Model-Based Approaches
Model-based methods for OPE rely on learning a transition model or approximating the state distribution to evaluate the target policy. These approaches can complement or replace IS techniques.

### 3.2.1 Transition Models for Evaluation
Transition models predict the next state $s'$ and reward $r$ given the current state-action pair $(s, a)$. A common approach is to fit a parametric model $p(s', r | s, a)$ using historical data. Once learned, this model can simulate trajectories under the target policy, enabling direct evaluation of its performance. However, inaccuracies in the model can propagate errors, leading to biased estimates.

### 3.2.2 State Distribution Approximation
State distribution approximation involves estimating the discrepancy between the stationary distributions induced by the behavior and target policies. This is often achieved through density ratio estimation techniques. For instance, the method proposed by [Swaminathan et al., 2015] uses regression to approximate the ratio of densities, which can then be used to correct biases in the estimated returns.

## 3.3 Doubly Robust Estimators
Doubly robust (DR) estimators combine IS techniques with model-based predictions to achieve lower variance and bias. They leverage the strengths of both approaches while mitigating their weaknesses.

### 3.3.1 Combining Importance Sampling and Model-Based Methods
DR estimators use a weighted combination of IS and model-based predictions. The general form of a DR estimator is:
$$
V^{\text{DR}}(s) = \mathbb{E}_{a \sim \mu}[\frac{\pi(a|s)}{\mu(a|s)} Q^{\mu}(s, a)] + \mathbb{E}_{a \sim \pi}[A^{\mu}(s, a)],
$$
where $Q^{\mu}$ is the action-value function under the behavior policy, and $A^{\mu}$ is the advantage function. If either the IS weights or the model is accurate, the DR estimator remains consistent.

### 3.3.2 Variance Reduction Techniques
To further improve the efficiency of DR estimators, variance reduction techniques such as control variates or truncation of IS weights are employed. These methods aim to stabilize the estimates without compromising accuracy.

## 3.4 Recent Advances
Recent research has expanded the scope of OPE by integrating neural networks and applying it to real-world domains like robotics and healthcare.

### 3.4.1 Neural Network-Based Estimators
Neural networks have been increasingly utilized to learn flexible representations for OPE. For example, Fitted-Q Evaluation (FQE) leverages neural networks to approximate the action-value function $Q^{\pi}$, providing a scalable solution for large state spaces. Additionally, methods like Retrace($\lambda$) incorporate eligibility traces to balance bias and variance.

### 3.4.2 Focused Applications in Robotics and Healthcare
In robotics, OPE is crucial for evaluating policies learned in simulation before deployment in real-world environments. Similarly, in healthcare, OPE enables the assessment of treatment policies using observational data. Both domains benefit from advancements in efficient and robust OPE algorithms.

![](placeholder_for_figure)

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Basic IS | Simple and unbiased | High variance |
| PDIS | Reduced variance | Still sensitive to long horizons |
| SIS | Stable estimates | Requires knowledge of stationary distribution |

# 4 Comparative Analysis

In this section, we provide a comparative analysis of the various off-policy evaluation (OPE) methods discussed in the literature review. The focus is on understanding their strengths and limitations, as well as analyzing empirical studies conducted to benchmark these methods.

## 4.1 Strengths and Limitations of Different Methods

To effectively evaluate policies without interacting with the environment, it is crucial to understand the trade-offs inherent in different OPE techniques. Below, we delve into two critical aspects: bias-variance trade-offs and computational complexity.

### 4.1.1 Bias-Variance Trade-offs

A fundamental challenge in OPE is balancing bias and variance. Importance sampling (IS) methods, such as basic IS and per-decision IS, are unbiased estimators but often suffer from high variance, especially when the behavior policy differs significantly from the target policy. This variance issue can be mitigated by using stationary importance sampling or doubly robust estimators, which combine model-based predictions with IS corrections. Mathematically, the variance of an IS estimator can be expressed as:

$$
\text{Var}(\hat{V}_{\text{IS}}) = \mathbb{E}\left[\left(\frac{\pi(a|s)}{\mu(a|s)}\right)^2 A^2\right] - \mathbb{E}[\hat{V}_{\text{IS}}]^2,
$$
where $\pi(a|s)$ and $\mu(a|s)$ represent the target and behavior policies, respectively, and $A$ denotes the advantage function.

Doubly robust estimators aim to reduce variance by leveraging both IS and model-based components, making them more robust to model misspecification. However, they may introduce additional bias if the model is inaccurate.

### 4.1.2 Computational Complexity

The computational cost of OPE methods varies widely depending on their design. For instance, basic IS requires only a single pass through the logged data, making it computationally efficient. In contrast, model-based approaches necessitate learning transition dynamics, which can be computationally intensive, particularly in high-dimensional state spaces. Neural network-based estimators further exacerbate this issue due to the need for training deep models.

| Method | Bias | Variance | Computational Complexity |
|--------|------|----------|-------------------------|
| Basic IS | High | High | Low |
| Doubly Robust | Moderate | Moderate | Moderate |
| Model-Based | Low | Low | High |

## 4.2 Empirical Studies and Benchmarks

Empirical evaluations play a pivotal role in assessing the practical performance of OPE methods. Below, we discuss standard benchmark environments and the metrics used to compare these methods.

### 4.2.1 Standard Benchmark Environments

Several environments have been established as benchmarks for evaluating OPE methods. These include classic control tasks like CartPole and MountainCar, as well as more complex domains such as Mujoco robotics simulations. Additionally, real-world datasets from healthcare and recommendation systems provide valuable testbeds for assessing the applicability of OPE methods in practical scenarios.

![](placeholder_for_benchmark_environments)

### 4.2.2 Performance Metrics Used

Performance metrics for OPE methods typically include absolute error, root mean squared error (RMSE), and normalized discounted cumulative gain (NDCG). These metrics help quantify the accuracy of estimated policy values relative to ground truth. For example, RMSE is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{V}_i - V_i)^2},
$$
where $\hat{V}_i$ and $V_i$ denote the estimated and true values of the $i$-th policy, respectively.

Moreover, metrics specific to certain applications, such as precision-recall curves in healthcare settings, are also employed to evaluate the effectiveness of OPE methods in specialized domains.

# 5 Discussion

In this section, we delve into the open challenges and future directions in off-policy evaluation (OPE), a critical component of reinforcement learning. While significant progress has been made in developing OPE methods, several key issues remain unresolved. Below, we discuss these challenges and outline promising avenues for future research.

## 5.1 Open Challenges in Off-Policy Evaluation

Off-policy evaluation faces numerous challenges that hinder its applicability to real-world problems. Two major challenges are handling high-dimensional state spaces and dealing with partial observability.

### 5.1.1 Handling High-Dimensional State Spaces

High-dimensional state spaces pose a significant challenge for OPE methods due to the curse of dimensionality. Traditional importance sampling techniques often suffer from high variance when applied to such settings. For example, in continuous control tasks, the state space can be infinite, making it computationally infeasible to estimate the value function accurately. To mitigate this issue, researchers have explored neural network-based estimators, which approximate the value function using deep learning architectures. However, these approaches introduce additional biases, necessitating careful trade-offs between bias and variance.

$$
\text{Variance of Importance Sampling: } \mathbb{V}[W] = \mathbb{E}[W^2] - (\mathbb{E}[W])^2,
$$
where $W$ is the importance weight. In high-dimensional spaces, the variance of $W$ grows exponentially, leading to unreliable estimates.

![](placeholder_for_high_dimensional_state_space_figure)

### 5.1.2 Dealing with Partial Observability

Partial observability arises when the agent does not have access to the full state information, leading to a partially observable Markov decision process (POMDP). In such scenarios, standard OPE methods fail because they assume the environment is fully observable. Techniques like latent variable models or recurrent neural networks (RNNs) have been proposed to address this issue by inferring hidden states. Nevertheless, these methods often require large amounts of data and computational resources, limiting their practicality.

| Challenge | Description |
|----------|-------------|
| High-Dimensional State Spaces | Exponential growth in variance for importance sampling. |
| Partial Observability | Difficulty in estimating value functions without full state information. |

## 5.2 Future Directions

To overcome the challenges discussed above, several promising future directions exist for advancing off-policy evaluation.

### 5.2.1 Integration with Online Learning

One potential direction is integrating OPE with online learning frameworks. By combining offline evaluation with online adaptation, agents can continuously refine their policies while maintaining robustness to distribution shifts. For instance, methods such as adaptive importance sampling dynamically adjust the proposal distribution based on incoming data, reducing variance over time. This integration could lead to more efficient and accurate policy evaluations in dynamic environments.

### 5.2.2 Developing More Efficient Algorithms

Efficiency remains a key concern for OPE algorithms, especially in large-scale applications. Developing algorithms that reduce both computational complexity and memory requirements is essential. Recent advances in mini-batch processing and parallel computation offer opportunities to scale up OPE methods. Additionally, exploring hybrid approaches that combine model-based and model-free techniques may yield more efficient solutions. For example, doubly robust estimators leverage both importance sampling and model-based predictions to achieve lower variance.

$$
\text{Doubly Robust Estimator: } Q_{\text{DR}}(s, a) = \hat{Q}(s, a) + \sum_{t=1}^T \rho_t (r_t + \gamma \hat{V}(s_{t+1}) - \hat{Q}(s_t, a_t)),
$$
where $\rho_t$ is the importance weight at time $t$, and $\hat{Q}$ and $\hat{V}$ are learned value functions.

# 6 Conclusion

In this survey, we have explored the critical topic of off-policy evaluation (OPE) in reinforcement learning, delving into its theoretical foundations, methodologies, and practical applications. This concluding section synthesizes the key findings from the preceding sections and discusses their implications for real-world scenarios.

## 6.1 Summary of Key Findings

Off-policy evaluation is a cornerstone of reinforcement learning, enabling the assessment of policies using data generated by different behavioral policies. The importance of OPE lies in its ability to facilitate safe and efficient policy improvement without requiring interaction with the environment during evaluation. Below, we summarize the main insights:

1. **Problem Definition and Importance**: Off-policy evaluation addresses the challenge of estimating the performance of a target policy using historical data collected under a different behavior policy. This capability is crucial in domains where direct experimentation is costly or risky, such as healthcare and robotics.

2. **Importance Sampling Techniques**: Methods like basic importance sampling (IS), per-decision IS, and stationary IS provide unbiased estimators but often suffer from high variance, especially in long-horizon problems. Variance reduction techniques, such as truncation and regularization, are essential for improving estimator stability.

3. **Model-Based Approaches**: These methods approximate the dynamics of the environment to simulate trajectories under the target policy. While they can reduce variance compared to IS-based methods, their accuracy depends heavily on the quality of the learned transition model.

4. **Doubly Robust Estimators**: By combining IS and model-based approaches, doubly robust estimators achieve a balance between bias and variance. Their effectiveness relies on accurate value function estimation and transition modeling.

5. **Recent Advances**: Neural network-based estimators and domain-specific applications (e.g., in robotics and healthcare) demonstrate the growing sophistication of OPE techniques. These advances underscore the potential for integrating deep learning with traditional OPE methods.

6. **Challenges and Open Problems**: High-dimensional state spaces, partial observability, and computational complexity remain significant hurdles. Addressing these challenges requires innovative algorithmic designs and theoretical breakthroughs.

| Key Aspect | Strengths | Limitations |
|------------|-----------|-------------|
| Importance Sampling | Unbiased estimates | High variance |
| Model-Based | Low variance | Model inaccuracies |
| Doubly Robust | Balanced bias-variance trade-off | Requires accurate models and value functions |

## 6.2 Implications for Practical Applications

The advancements in off-policy evaluation have profound implications for practical applications across various domains. For instance:

- **Healthcare**: In personalized medicine, OPE allows for the evaluation of treatment policies using observational data, minimizing the need for controlled experiments that could be ethically challenging.

- **Robotics**: Autonomous systems benefit from OPE by enabling offline policy evaluation, reducing the reliance on expensive and time-consuming physical trials.

- **Recommendation Systems**: OPE techniques help assess new recommendation strategies using logged user interactions, ensuring that changes do not degrade user experience.

However, translating these theoretical insights into practice requires careful consideration of domain-specific constraints. For example, in safety-critical applications like autonomous driving, the robustness and reliability of OPE methods must be rigorously validated. Additionally, the computational demands of modern OPE techniques necessitate scalable implementations, particularly when dealing with large datasets or complex environments.

As research progresses, the integration of OPE with online learning and the development of more efficient algorithms will further enhance its applicability. These future directions hold promise for expanding the scope and impact of off-policy evaluation in reinforcement learning.

