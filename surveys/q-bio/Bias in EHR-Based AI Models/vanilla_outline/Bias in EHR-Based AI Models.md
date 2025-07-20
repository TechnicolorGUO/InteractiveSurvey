# 1 Introduction

The integration of artificial intelligence (AI) into healthcare systems has the potential to revolutionize patient care, streamline operations, and improve clinical outcomes. A significant portion of this progress relies on electronic health records (EHRs), which serve as a foundational data source for AI models. However, the reliance on EHR data introduces unique challenges, particularly in the form of biases that can compromise model fairness, accuracy, and reliability. This survey explores the multifaceted issue of bias in EHR-based AI models, examining its origins, implications, and mitigation strategies.

## 1.1 Background on EHR-Based AI Models

Electronic health records (EHRs) are digital repositories of patient information, including demographics, medical history, diagnoses, treatments, and laboratory results. These rich datasets have become invaluable for training AI models aimed at predicting patient outcomes, identifying disease patterns, and optimizing treatment plans. Machine learning algorithms, such as supervised learning, deep learning, and natural language processing (NLP), leverage EHR data to uncover insights that may not be apparent through traditional statistical methods. For instance, predictive models built on EHR data can estimate readmission risks or detect early signs of sepsis.

However, the quality and representativeness of EHR data directly influence the performance and fairness of AI models. Issues such as missing data, inconsistent documentation practices, and demographic imbalances can lead to biased predictions. The mathematical foundation of these models often assumes that the training data is an unbiased sample of the population, but in practice, this assumption is frequently violated. For example, if certain demographic groups are underrepresented in the dataset, the model's predictions for those groups may suffer from higher error rates.

$$
\text{Model Prediction Error} = f(\text{Data Quality}, \text{Bias in Training Data})
$$

## 1.2 Importance of Addressing Bias

Bias in EHR-based AI models poses significant ethical, legal, and practical challenges. From an ethical standpoint, biased models can perpetuate existing inequalities by disproportionately affecting marginalized populations. For instance, a model trained on predominantly male patient data may fail to accurately predict outcomes for female patients. Legally, the use of biased AI systems in healthcare could result in liability issues, as regulatory bodies increasingly scrutinize the fairness and transparency of AI-driven decisions. Practically, biased models reduce overall system reliability, leading to suboptimal patient care and diminished trust in AI technologies.

To address these concerns, it is essential to identify and quantify the sources of bias in EHR data and develop robust mitigation strategies. This involves not only technical solutions but also interdisciplinary collaboration between data scientists, clinicians, ethicists, and policymakers.

## 1.3 Objectives of the Survey

This survey aims to provide a comprehensive overview of the challenges and opportunities associated with bias in EHR-based AI models. Specifically, the objectives are as follows:

1. To identify and categorize the primary sources of bias in EHR data, including demographic, clinical documentation, and algorithmic biases.
2. To review existing approaches for mitigating bias, ranging from preprocessing techniques to model-level interventions and postprocessing adjustments.
3. To evaluate metrics and methodologies for assessing bias in AI models, both quantitatively and qualitatively.
4. To discuss the ethical considerations surrounding the deployment of EHR-based AI systems, emphasizing the need for transparency, accountability, and governance.
5. To highlight current gaps in research and propose future directions for advancing the field.

By addressing these objectives, this survey seeks to inform researchers, practitioners, and policymakers about the complexities of bias in EHR-based AI models and guide the development of fairer and more equitable healthcare systems.

# 2 Literature Review Framework

In this section, we establish the framework for the literature review, defining the scope of our analysis and detailing the methodology used to select relevant studies. This structured approach ensures a comprehensive understanding of bias in Electronic Health Record (EHR)-based AI models.

## 2.1 Scope and Definitions

The scope of this survey encompasses the identification, characterization, and mitigation of biases in EHR-based AI models. These models leverage large datasets derived from patient records to predict outcomes, diagnose conditions, or guide clinical decisions. However, inherent biases in the data can lead to unfair or inaccurate predictions, particularly affecting underrepresented populations.

### Key Definitions
- **Bias**: A systematic error in the data or model that leads to skewed predictions favoring or disfavoring specific groups.
- **EHR Data**: Structured and unstructured health information collected during routine patient care.
- **AI Models**: Algorithms trained on EHR data to perform predictive or classification tasks.

We focus on three primary types of bias: demographic, clinical documentation, and algorithmic. Each type is explored in subsequent sections, with an emphasis on their origins and implications for healthcare equity.

| Type of Bias       | Description                                                                 |
|--------------------|---------------------------------------------------------------------------|
| Demographic Bias   | Unequal representation of certain population groups in the dataset.         |
| Clinical Bias      | Variability in how health information is recorded across different contexts.|
| Algorithmic Bias   | Flaws in the model architecture or training process leading to unfairness.  |

## 2.2 Methodology for Selection of Studies

To ensure the robustness of our findings, a systematic methodology was employed to select studies for inclusion in this survey. The process involved the following steps:

1. **Database Selection**: We searched major databases such as PubMed, IEEE Xplore, and ACM Digital Library using keywords like "bias in EHR," "AI fairness," and "healthcare disparities."
2. **Inclusion Criteria**: Studies were included if they addressed bias in EHR-based AI models, provided empirical evidence, and were published within the last decade.
3. **Exclusion Criteria**: Reviews, editorials, and studies focusing solely on non-EHR datasets were excluded.
4. **Quality Assessment**: Each study underwent a quality assessment based on its methodology, sample size, and relevance to the topic.

$$
N_{\text{included}} = \sum_{i=1}^{M} \mathbb{1}_{\text{meets criteria}}(S_i)
$$
where $N_{\text{included}}$ represents the number of studies included, $M$ is the total number of retrieved studies, and $\mathbb{1}_{\text{meets criteria}}(S_i)$ is an indicator function evaluating whether study $S_i$ satisfies the inclusion criteria.

A flowchart summarizing the selection process is shown below:

![]()

This rigorous methodology ensures that the reviewed literature provides a balanced and representative view of the current state of research on bias in EHR-based AI models.

# 3 Sources of Bias in EHR Data

Electronic Health Records (EHRs) serve as the backbone for many AI models in healthcare, but they are not immune to biases that can compromise model fairness and accuracy. This section explores the primary sources of bias in EHR data, categorized into demographic bias, clinical documentation bias, and algorithmic bias.

## 3.1 Demographic Bias

Demographic bias arises when certain population groups are underrepresented or misrepresented in EHR datasets. This can lead to AI models that perform poorly for these groups, exacerbating existing health disparities.

### 3.1.1 Underrepresentation of Minorities

Minority populations often experience lower rates of healthcare utilization, resulting in fewer records being captured in EHR systems. For instance, studies have shown that racial minorities may constitute only a fraction of the dataset used to train predictive models, leading to biased predictions. The disparity can be quantified using metrics such as the proportion of minority patients relative to their representation in the general population:
$$	ext{Underrepresentation Ratio} = \frac{\text{Proportion of Minority Patients in Dataset}}{\text{Proportion of Minority Patients in Population}}.$$
This ratio below 1 indicates underrepresentation. Addressing this issue requires deliberate efforts to collect more inclusive data.

### 3.1.2 Gender Disparities

Gender disparities in EHR data manifest through uneven distributions of male and female patients across conditions or treatments. For example, certain diseases might predominantly affect one gender, leading to skewed datasets. Additionally, differences in how symptoms are documented between genders can introduce further bias. A table summarizing gender distribution across various conditions could help illustrate this point:

| Condition | Male (%) | Female (%) |
|----------|-----------|------------|
| Diabetes | 45        | 55         |
| Heart Disease | 60       | 40         |

Such imbalances necessitate careful preprocessing to ensure fair model performance.

## 3.2 Clinical Documentation Bias

Clinical documentation bias occurs due to inconsistencies and subjectivity in how patient information is recorded within EHRs.

### 3.2.1 Inconsistent Recording Practices

Healthcare providers may vary in how they document patient encounters, leading to incomplete or inconsistent data entries. For example, some practitioners might omit critical details about social determinants of health, while others may overemphasize certain aspects. Standardization of documentation practices is essential to mitigate this form of bias.

### 3.2.2 Subjectivity in Notes

Free-text notes in EHRs often reflect the subjective interpretations of clinicians. These notes can encode implicit biases, such as stereotypical assumptions about patient behavior based on race or socioeconomic status. Natural Language Processing (NLP) techniques must account for these nuances to avoid perpetuating such biases in downstream applications.

## 3.3 Algorithmic Bias

Algorithmic bias stems from issues inherent in the design and training of AI models, which can amplify biases present in the underlying EHR data.

### 3.3.1 Training Data Imbalance

Training data imbalance occurs when certain classes or groups dominate the dataset, causing models to favor those groups during prediction. Mathematically, this can be represented as an imbalance in class proportions:
$$P(\text{Class}_i) \gg P(\text{Class}_j),$$
where $P(\text{Class}_i)$ represents the probability of observing a particular class. Techniques like oversampling or undersampling can address this issue, though they come with trade-offs in terms of computational cost and model complexity.

### 3.3.2 Model Architecture Limitations

The architecture of AI models themselves can introduce bias if they fail to adequately capture complex relationships in the data. For example, simpler models may struggle to generalize across diverse populations, while deep learning models may require prohibitively large amounts of balanced data to achieve fairness. ![](placeholder_for_model_architecture_diagram) A diagram comparing different architectures' susceptibility to bias could provide additional clarity.

# 4 Approaches to Mitigate Bias

Addressing bias in electronic health record (EHR)-based AI models is critical for ensuring fairness, accuracy, and reliability in healthcare applications. This section explores various approaches to mitigate bias at different stages of the AI pipeline: preprocessing, model-level interventions, and postprocessing adjustments.

## 4.1 Preprocessing Techniques

Preprocessing techniques aim to reduce bias by modifying the input data before it is fed into an AI model. These methods focus on improving the quality and representativeness of the dataset.

### 4.1.1 Data Augmentation

Data augmentation involves generating synthetic samples to enrich underrepresented groups in the dataset. Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are commonly used to balance class distributions. For example, SMOTE creates new instances by interpolating between existing minority samples:
$$
x_{\text{new}} = x_i + \lambda(x_j - x_i),
$$
where $x_i$ and $x_j$ are two minority samples, and $\lambda$ is a random value between 0 and 1. While effective, care must be taken to ensure that augmented data remains clinically meaningful.

### 4.1.2 Rebalancing Datasets

Rebalancing datasets involves either oversampling minority classes or undersampling majority classes. Oversampling can preserve all original data but risks overfitting, while undersampling may lead to loss of valuable information. A hybrid approach combining both strategies often achieves better results. The choice of rebalancing technique depends on the specific characteristics of the dataset and the application domain.

## 4.2 Model-Level Interventions

Model-level interventions address bias by incorporating fairness constraints directly into the training process. These methods modify the model architecture or learning algorithm to promote fairness.

### 4.2.1 Regularization Methods

Regularization methods penalize biased predictions during training. For instance, adding a fairness term to the loss function can encourage the model to produce equitable outcomes across demographic groups. A common formulation is:
$$
L = L_{\text{base}} + \lambda L_{\text{fairness}},
$$
where $L_{\text{base}}$ is the standard loss function, $L_{\text{fairness}}$ measures the degree of bias, and $\lambda$ controls the trade-off between accuracy and fairness.

### 4.2.2 Fairness-Constrained Optimization

Fairness-constrained optimization explicitly enforces fairness constraints during training. For example, equality of opportunity can be enforced by requiring that true positive rates are equal across groups:
$$
P(\hat{y} = 1 | y = 1, G = g) = P(\hat{y} = 1 | y = 1, G = g'),
$$
where $G$ denotes the group membership, and $g$ and $g'$ represent different groups. Solving such constrained optimization problems typically requires advanced algorithms like Lagrange multipliers or dual ascent.

## 4.3 Postprocessing Adjustments

Postprocessing techniques adjust the output of a trained model to mitigate bias without altering its internal parameters. These methods are particularly useful when retraining is not feasible.

### 4.3.1 Calibration of Predictions

Calibration ensures that predicted probabilities align with observed frequencies. Group-specific calibration can help address disparities in prediction accuracy across demographic groups. Techniques such as Platt scaling or isotonic regression can be applied separately for each group to achieve balanced performance.

| Method        | Description                                                                 |
|---------------|---------------------------------------------------------------------------|
| Platt Scaling | Fits a logistic regression model to map raw scores to calibrated probabilities. |
| Isotonic Regression | Uses a piecewise constant function to ensure monotonicity in calibration. |

### 4.3.2 Outcome Adjustment Strategies

Outcome adjustment modifies final decisions based on fairness considerations. For example, threshold optimization adjusts decision thresholds for different groups to achieve desired fairness metrics. Let $t_g$ denote the threshold for group $g$. The optimal thresholds can be determined by solving:
$$
\min_{t_g} \sum_g \text{Loss}(t_g) \quad \text{subject to fairness constraints.}
$$
This approach allows fine-grained control over fairness-accuracy trade-offs.

![](placeholder_for_figure.png)

In summary, mitigating bias in EHR-based AI models requires a multifaceted approach that integrates preprocessing, model-level interventions, and postprocessing techniques. Each method has its strengths and limitations, and their effectiveness depends on the specific context and goals of the application.

# 5 Evaluation Metrics for Bias Assessment

The evaluation of bias in EHR-based AI models is a critical step in ensuring fairness and reliability. This section explores both quantitative and qualitative metrics that are commonly used to assess bias in such models.

## 5.1 Quantitative Metrics
Quantitative metrics provide an objective way to measure the extent of bias in AI models. These metrics often rely on statistical comparisons between different demographic groups or subpopulations within the data.

### 5.1.1 Statistical Parity Difference
Statistical parity difference measures the disparity in outcomes between protected and unprotected groups. It is defined as the absolute difference in the probability of a positive prediction for two groups, $A$ and $B$, given by:

$$
SPD = |P(\hat{Y}=1|G=A) - P(\hat{Y}=1|G=B)|
$$

Where $\hat{Y}$ represents the predicted outcome and $G$ denotes the group membership. A value of $SPD = 0$ indicates no disparity in predictions across the groups. However, achieving statistical parity may not always align with other fairness criteria, leading to trade-offs in model performance.

### 5.1.2 Equal Opportunity Difference
Equal opportunity difference focuses on reducing false negatives among disadvantaged groups. It evaluates whether individuals from different groups who truly belong to the positive class are equally likely to receive a positive prediction. Mathematically, it is expressed as:

$$
EOD = |P(\hat{Y}=1|Y=1, G=A) - P(\hat{Y}=1|Y=1, G=B)|
$$

Here, $Y$ represents the true label. Minimizing $EOD$ ensures that qualified individuals from all groups have similar chances of being correctly identified.

## 5.2 Qualitative Analysis
While quantitative metrics offer precise measurements, they do not capture the full complexity of bias in real-world applications. Qualitative analysis complements these metrics by providing deeper insights into the impact of bias.

### 5.2.1 Case Studies in Healthcare
Case studies illustrate how bias manifests in specific healthcare contexts. For example, a study might examine the performance of an EHR-based AI model in predicting readmission rates across racial groups. Such analyses often reveal hidden biases that quantitative metrics alone cannot detect. ![](placeholder_for_case_study_diagram)

### 5.2.2 Stakeholder Feedback
Engaging stakeholders, including clinicians, patients, and policymakers, provides valuable perspectives on the implications of bias. Their feedback can highlight ethical concerns, practical limitations, and potential solutions. Structured interviews or surveys can be used to gather this input, as shown in the following table placeholder:

| Stakeholder Group | Key Concerns Raised |
|-------------------|---------------------|
| Clinicians        | Model transparency   |
| Patients          | Fairness in outcomes|
| Policymakers      | Regulatory compliance|

# 6 Ethical Considerations

The ethical implications of bias in electronic health record (EHR)-based AI models are profound, as these systems increasingly influence clinical decision-making. This section explores the dual pillars of transparency and accountability in ensuring that EHR-based AI systems are ethically sound.

## 6.1 Transparency in AI Systems

Transparency is a cornerstone of trust in AI systems, particularly in healthcare where decisions can have life-altering consequences. Ensuring that these systems are understandable and interpretable is critical for both clinicians and patients.

### 6.1.1 Explainability of EHR-Based Models

Explainability refers to the ability to understand why an AI model made a particular prediction or recommendation. For EHR-based models, this involves unpacking how features derived from patient data contribute to outcomes. Techniques such as SHAP (SHapley Additive exPlanations) values $\phi_i$ provide insights into feature importance:

$$
\phi_i = \frac{1}{M!} \sum_{S \subseteq N \setminus \{i\}} \binom{M-1}{|S|}^{-1} [f(S \cup \{i\}) - f(S)]
$$

where $f(S)$ represents the model output given a subset $S$ of features. While SHAP and similar methods enhance explainability, they often come at the cost of computational complexity. Additionally, the interpretability of complex models like deep neural networks remains a significant challenge.

### 6.1.2 Audit Trails for Decision-Making

Audit trails document the reasoning process behind AI-generated outputs, enabling retrospective analysis and accountability. These logs should capture not only the final decision but also intermediate steps, including input data, model parameters, and any preprocessing applied. An effective audit trail system must balance granularity with usability, ensuring that it provides meaningful insights without overwhelming users. ![]()

## 6.2 Accountability and Governance

Accountability ensures that stakeholders—developers, clinicians, and regulators—are held responsible for the ethical deployment of AI systems. This subsection examines regulatory frameworks and liability considerations.

### 6.2.1 Regulatory Standards

Regulatory bodies play a pivotal role in establishing guidelines for the development and deployment of AI in healthcare. For instance, the FDA’s Software as a Medical Device (SaMD) framework emphasizes premarket evaluation and postmarket monitoring. Table 1 summarizes key regulatory standards across jurisdictions.

| Jurisdiction | Standard | Focus |
|-------------|----------|-------|
| United States | FDA SaMD | Safety and efficacy |
| European Union | GDPR | Data protection |
| Canada | Health Canada | Risk management |

### 6.2.2 Liability Frameworks

Liability frameworks address who bears responsibility when an AI-driven decision leads to harm. In EHR-based models, potential liable parties include developers, hospitals, and even individual clinicians. A probabilistic approach to liability could involve quantifying the likelihood of harm attributable to each party. For example, if a biased model disproportionately affects minority groups, the developer might bear greater liability due to inadequate testing. However, legal precedents in this domain remain nascent, necessitating further exploration and clarification.

# 7 Discussion

In this section, we synthesize the findings from the preceding sections and highlight current gaps in research as well as potential future directions for addressing bias in EHR-based AI models.

## 7.1 Current Gaps in Research

Despite significant advancements in understanding and mitigating bias in EHR-based AI models, several critical gaps remain. First, there is a lack of standardized benchmarks for evaluating bias across different healthcare domains. While some studies propose metrics such as statistical parity difference ($SPD = P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)$) and equal opportunity difference ($EOD = P(\hat{Y}=1|Y=1, A=0) - P(\hat{Y}=1|Y=1, A=1)$), these are often applied inconsistently or incompletely. This inconsistency hinders meaningful comparisons between studies.

Second, the majority of existing research focuses on demographic biases (e.g., race and gender), leaving other forms of bias—such as socioeconomic status or geographic disparities—underexplored. For instance, while Section 3.1 discusses underrepresentation of minorities, less attention has been paid to how intersectional identities compound bias. Additionally, clinical documentation bias (Section 3.2) remains challenging to quantify due to its subjective nature, and more robust methods for detecting inconsistencies in recording practices are needed.

Finally, algorithmic bias mitigation techniques (Section 4) often prioritize theoretical fairness over practical applicability. Many proposed solutions assume access to large, balanced datasets or require computationally expensive retraining processes, which may not be feasible in resource-constrained settings. Furthermore, the trade-offs between fairness and model performance are not always adequately addressed, leading to suboptimal implementations in real-world scenarios.

| Gap Area | Description |
|----------|-------------|
| Standardized Metrics | Inconsistent application of bias evaluation metrics across studies. |
| Intersectional Bias | Limited exploration of compounded biases beyond demographics. |
| Practicality of Solutions | Theoretical fairness approaches that lack feasibility in clinical practice. |

## 7.2 Future Directions

To address these gaps, we outline several promising avenues for future research. One key direction involves developing domain-specific benchmarks tailored to the unique challenges of healthcare data. These benchmarks should incorporate both quantitative metrics (e.g., $SPD$, $EOD$) and qualitative assessments (e.g., stakeholder feedback, case studies). By creating shared standards, researchers can better evaluate the effectiveness of bias mitigation strategies across diverse populations and use cases.

Another important area is the integration of explainability tools into EHR-based AI systems. As discussed in Section 6.1, transparency is crucial for building trust among clinicians and patients. Future work could focus on designing interpretable models that provide actionable insights into predictions while maintaining high accuracy. For example, post-hoc explanation methods like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) could be adapted specifically for EHR data.

Moreover, interdisciplinary collaboration will play a vital role in advancing this field. Engaging ethicists, policymakers, and healthcare providers alongside data scientists can ensure that technical innovations align with societal values and regulatory requirements. Developing liability frameworks (Section 6.2) that clearly define responsibilities in cases of biased outcomes is particularly urgent, given the increasing deployment of AI in clinical decision-making.

Lastly, fostering inclusivity in dataset collection and model development is essential. Efforts should include expanding data representation through partnerships with underserved communities and leveraging synthetic data generation techniques to augment scarce samples. ![](placeholder_for_inclusivity_diagram)

By pursuing these directions, the scientific community can move closer to realizing equitable and reliable AI-driven healthcare solutions.

# 8 Conclusion

In this survey, we have explored the multifaceted issue of bias in Electronic Health Record (EHR)-based AI models. The following sections summarize the key findings and discuss their implications for practice.

## 8.1 Summary of Key Findings

The literature review has revealed several critical insights regarding the sources, mitigation strategies, and evaluation metrics for bias in EHR-based AI systems. First, bias originates from various levels within the data pipeline: demographic disparities lead to underrepresentation of minorities and gender imbalances, while clinical documentation practices introduce inconsistencies and subjectivity. Algorithmic bias further compounds these issues through training data imbalances and limitations in model architectures. 

To address these challenges, researchers have proposed a range of approaches at different stages of the AI pipeline. Preprocessing techniques such as data augmentation and rebalancing datasets help mitigate biases in the input data. At the model level, interventions like regularization methods and fairness-constrained optimization aim to reduce bias during training. Postprocessing adjustments, including calibration of predictions and outcome adjustment strategies, ensure that model outputs are equitable across diverse populations.

Evaluation metrics play a pivotal role in quantifying and addressing bias. Quantitative metrics such as statistical parity difference ($SPD = P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)$) and equal opportunity difference ($EOD = P(\hat{Y}=1|Y=1, A=0) - P(\hat{Y}=1|Y=1, A=1)$) provide objective measures of fairness. Qualitative analyses, including case studies and stakeholder feedback, complement these numerical assessments by offering real-world context.

Ethical considerations underscore the importance of transparency and accountability in AI systems. Explainability techniques enable stakeholders to understand model decisions, while audit trails ensure traceability of the decision-making process. Regulatory standards and liability frameworks further reinforce governance over AI applications in healthcare.

## 8.2 Implications for Practice

The findings of this survey carry significant implications for practitioners, policymakers, and researchers working with EHR-based AI models. Practitioners must adopt a holistic approach to bias mitigation, integrating preprocessing, model-level, and postprocessing techniques tailored to their specific use cases. For instance, medical institutions could implement data augmentation strategies to address underrepresented groups in their datasets or employ fairness-constrained optimization to train more equitable models.

Policymakers should prioritize the development of regulatory frameworks that enforce transparency and accountability in AI systems. This includes mandating explainability reports for high-stakes applications and establishing clear guidelines for auditing AI models. Furthermore, fostering collaboration between technical experts, clinicians, and ethicists can enhance the ethical deployment of AI in healthcare.

For researchers, the current gaps identified in this survey highlight promising avenues for future work. These include advancing novel methodologies for bias detection and mitigation, refining evaluation metrics to better capture nuanced forms of bias, and exploring the long-term impacts of biased AI on patient outcomes. Additionally, interdisciplinary research that incorporates perspectives from social sciences and humanities can enrich our understanding of the broader societal implications of AI in healthcare.

In conclusion, addressing bias in EHR-based AI models is a complex but essential endeavor. By synthesizing knowledge from multiple domains and adopting a proactive stance toward fairness and equity, we can pave the way for more reliable and trustworthy AI systems in healthcare.

