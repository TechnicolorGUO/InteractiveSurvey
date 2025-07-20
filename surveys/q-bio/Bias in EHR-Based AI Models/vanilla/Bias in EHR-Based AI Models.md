# Bias in EHR-Based AI Models: A Comprehensive Literature Survey

## Introduction
Electronic Health Records (EHRs) have become a cornerstone of modern healthcare systems, providing vast amounts of data that can be leveraged for artificial intelligence (AI) and machine learning (ML) applications. However, the use of EHR-based AI models has raised concerns about bias, which can lead to inequitable or harmful outcomes. This survey explores the sources, manifestations, and mitigation strategies for bias in EHR-based AI models.

## Sources of Bias in EHR Data

### Data Collection Biases
EHR data often reflects systemic biases present in healthcare delivery. These biases stem from:
- **Underrepresentation**: Certain demographic groups may not be adequately represented in EHR datasets. For instance, rural populations or minority groups might have fewer records due to disparities in healthcare access.
- **Measurement Error**: Differences in how data is collected across institutions can introduce inconsistencies. For example, some hospitals may record blood pressure differently than others.

$$	ext{Bias} = \frac{\text{Observed Distribution}}{\text{True Population Distribution}} - 1$$

This formula quantifies the degree of bias as the deviation from the true population distribution.

### Algorithmic Biases
Even when EHR data is unbiased, the algorithms used to process it can introduce bias. Common issues include:
- **Overfitting**: Models may overfit to specific subgroups within the training data, leading to poor generalization.
- **Feature Selection Bias**: Features chosen for model training may inadvertently encode societal biases.

![](placeholder_for_algorithmic_bias_diagram)

## Manifestations of Bias in EHR-Based AI Models

### Clinical Decision Support Systems
Bias in AI models used for clinical decision support can result in incorrect diagnoses or treatment recommendations. For example, a model trained predominantly on male patient data might fail to accurately predict conditions more prevalent in females.

| Condition | Male Prevalence (%) | Female Prevalence (%) |
|-----------|---------------------|-----------------------|
| Heart Disease | 45 | 30 |
| Autoimmune Disorders | 20 | 80 |

### Predictive Modeling
In predictive modeling, bias can manifest as unequal prediction accuracy across different demographic groups. For instance, a readmission risk model might underpredict risks for minority patients due to historical underdiagnosis in their records.

$$P(\text{Outcome} | \text{Demographic Group}) \neq P(\text{Outcome})$$

This inequality highlights the disparity in predicted probabilities across groups.

## Mitigation Strategies

### Data Preprocessing Techniques
To address bias at the data level, preprocessing techniques such as resampling and weighting can be employed. Resampling involves adjusting the dataset to balance representation across groups, while weighting assigns higher importance to underrepresented samples during training.

$$w_i = \frac{1}{N_c}, \quad \forall i \in \text{Group } c$$

Here, $w_i$ represents the weight assigned to sample $i$, and $N_c$ is the number of samples in group $c$.

### Fairness-Aware Algorithms
Fairness-aware algorithms explicitly incorporate fairness constraints into the model training process. Techniques include:
- **Adversarial Debiasing**: Training a secondary model to detect and counteract biases in the primary model.
- **Post-Hoc Adjustments**: Modifying model outputs to ensure fairness metrics are met.

$$\min_{\theta} \mathcal{L}(\theta) + \lambda \cdot \text{Fairness Penalty}(\theta)$$

The above optimization problem balances model performance ($\mathcal{L}$) with fairness constraints.

### Continuous Monitoring
Once deployed, EHR-based AI models must be continuously monitored for bias. This involves tracking key metrics such as:
- **Disparate Impact**: The ratio of positive predictions for one group compared to another.
- **Equality of Opportunity**: Ensuring equal true positive rates across groups.

$$\text{Disparate Impact} = \frac{P(\hat{y}=1 | G=1)}{P(\hat{y}=1 | G=0)}$$

## Conclusion
Bias in EHR-based AI models poses significant challenges to equitable healthcare delivery. By understanding its sources, manifestations, and implementing effective mitigation strategies, researchers and practitioners can develop more reliable and fair AI systems. Future work should focus on developing standardized benchmarks for evaluating fairness and expanding the scope of fairness considerations beyond traditional demographic factors.
