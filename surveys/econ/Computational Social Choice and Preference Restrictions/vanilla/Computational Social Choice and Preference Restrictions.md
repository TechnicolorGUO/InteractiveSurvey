# Computational Social Choice and Preference Restrictions

## Introduction

Computational social choice (ComSoc) is an interdisciplinary field that combines tools from computer science, economics, and mathematics to study collective decision-making processes. A central problem in this domain is aggregating individual preferences into a collective decision while ensuring computational feasibility and fairness. However, unrestricted preference domains often lead to intractable problems or undesirable outcomes such as Arrow's impossibility theorem. To address these challenges, researchers have explored **preference restrictions**, which impose structural constraints on the set of allowable preferences. This survey provides an overview of key results and techniques in computational social choice under preference restrictions.

## Background and Key Concepts

### Social Choice Theory
Social choice theory studies how individual preferences can be aggregated into a collective decision. The primary goal is to design voting rules or aggregation mechanisms that satisfy desirable axiomatic properties such as fairness, strategy-proofness, and efficiency. 

#### Voting Rules
A voting rule $ f $ maps a profile of individual preferences $ P = (P_1, P_2, \ldots, P_n) $ into a collective outcome. Common voting rules include:
- **Plurality**: Each voter selects their top candidate, and the candidate with the most votes wins.
- **Borda Count**: Candidates are assigned scores based on their rank in each voter's preference list.
- **Single Transferable Vote (STV)**: An iterative elimination process where the candidate with the fewest votes is removed until a winner emerges.

### Preference Domains
Preferences are typically represented as total orders over a set of alternatives $ A $. For example, if $ A = \{a, b, c\} $, a voter might prefer $ a > b > c $. However, unrestricted preference domains can lead to computational hardness or impossibility results. Preference restrictions aim to simplify the domain by imposing constraints such as single-peakedness, single-crossing, or group-separable preferences.

## Preference Restrictions in Computational Social Choice

### Single-Peaked Preferences
Single-peaked preferences assume that alternatives can be arranged along a one-dimensional axis, and each voter has a most-preferred alternative with decreasing utility as you move away from it. Formally, for a linear order $ L $ over $ A $, a preference $ P_i $ is single-peaked if there exists a peak $ p_i \in A $ such that for all $ x, y \in A $,
$$
|x - p_i| < |y - p_i| \implies x \succ_i y.
$$

#### Computational Implications
Single-peaked preferences significantly reduce the complexity of many social choice problems. For instance, finding a Condorcet winner (an alternative that beats all others in pairwise comparisons) can be done efficiently under single-peaked preferences, whereas it is NP-hard in general domains.

![](single_peaked_preferences_diagram_placeholder)

### Single-Crossing Preferences
Single-crossing preferences generalize single-peakedness by assuming that voters' preferences "cross" at most once when ordered along a line. This property arises naturally in settings where voters have ideological positions, and candidates are aligned along a spectrum.

| Property            | Single-Peaked | Single-Crossing |
|---------------------|---------------|------------------|
| Dimensionality      | 1D            | 1D               |
| Complexity Reduction| Significant   | Moderate         |

### Group-Separable Preferences
Group-separable preferences allow voters to partition alternatives into groups such that preferences within each group are independent of those outside it. This restriction is useful in multi-issue elections or committee selection problems.

$$
P_i \text{ is group-separable if } \forall G \subseteq A, \exists R_G \text{ s.t. } R_G \text{ ranks } G \text{ independently.}
$$

#### Applications
Group-separable preferences are particularly relevant in combinatorial voting scenarios, such as selecting committees or bundles of goods. Algorithms leveraging these restrictions can achieve polynomial-time solutions for problems that are otherwise computationally hard.

## Algorithmic Techniques Under Preference Restrictions

### Efficient Aggregation Algorithms
Preference restrictions enable the design of efficient algorithms for aggregating preferences. For example, under single-peaked preferences, the median voter theorem guarantees that the median voter's preferred alternative is a Condorcet winner. Similarly, dynamic programming techniques can be used to compute winners under scoring rules like Borda count.

### Complexity Analysis
While many problems in unrestricted domains are NP-hard, preference restrictions often reduce their complexity to polynomial time. Table below summarizes the computational complexity of various problems under different restrictions.

| Problem                  | Unrestricted | Single-Peaked | Single-Crossing |
|-------------------------|--------------|---------------|------------------|
| Winner Determination    | NP-hard      | Polynomial    | Polynomial       |
| Coalitional Manipulation| NP-hard      | Polynomial    | Open Question    |

## Conclusion

Preference restrictions play a crucial role in making computational social choice problems tractable while preserving meaningful axiomatic properties. Single-peaked, single-crossing, and group-separable preferences are among the most widely studied restrictions, each offering unique advantages in specific contexts. Future research may explore hybrid preference restrictions or develop new algorithmic techniques tailored to emerging applications in AI-driven decision-making systems.
