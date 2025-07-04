# AI in Schizophrenia Diagnosis via EEG: A Literature Survey

## Introduction
Schizophrenia is a complex psychiatric disorder characterized by disruptions in thought processes, perceptions, emotional responsiveness, and social interactions. Early and accurate diagnosis is critical for effective treatment and management of the condition. Electroencephalography (EEG) has emerged as a promising tool for studying brain activity in schizophrenia due to its high temporal resolution and non-invasive nature. Recent advancements in artificial intelligence (AI), particularly machine learning (ML) and deep learning (DL), have enabled the development of sophisticated models capable of analyzing EEG data to identify biomarkers associated with schizophrenia. This survey explores the current state of research on AI-based approaches for diagnosing schizophrenia using EEG.

## Background
### Schizophrenia and EEG
Schizophrenia is often associated with abnormalities in neural oscillations, which can be captured through EEG recordings. These abnormalities include alterations in power spectra, coherence, and phase synchronization across various frequency bands (e.g., delta, theta, alpha, beta, and gamma). Understanding these changes provides insights into the neurophysiological underpinnings of the disorder.

### Role of AI in Medical Diagnosis
AI techniques, especially ML and DL, have revolutionized medical diagnostics by enabling automated feature extraction and classification from complex datasets. In the context of schizophrenia, AI models can process large volumes of EEG data to detect subtle patterns that may elude human analysts.

## Methods and Techniques
### Preprocessing EEG Data
Before applying AI algorithms, raw EEG signals undergo preprocessing steps such as filtering, artifact removal, and segmentation. Commonly used filters include bandpass filters to isolate specific frequency bands and notch filters to remove noise (e.g., 50/60 Hz line interference).

$$	ext{Filtered Signal} = H(f) \times X(f)$$
where $H(f)$ represents the filter transfer function and $X(f)$ is the Fourier transform of the raw signal.

### Feature Extraction
Feature extraction involves identifying relevant characteristics of the EEG data that can distinguish between healthy controls and individuals with schizophrenia. Traditional methods rely on handcrafted features such as spectral power, entropy, and coherence. Modern approaches leverage automated feature learning through DL architectures like convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

| Feature Type | Description |
|-------------|-------------|
| Spectral Power | Measures energy distribution across frequency bands. |
| Coherence | Quantifies synchronization between different brain regions. |
| Entropy | Captures complexity and irregularity in EEG signals. |

### Classification Algorithms
Several ML and DL algorithms have been employed for schizophrenia diagnosis using EEG data. Popular techniques include:

- **Support Vector Machines (SVM):** Effective for binary classification tasks.
- **Random Forests:** Robust to overfitting and capable of handling high-dimensional data.
- **Deep Learning Models:** CNNs excel at spatial pattern recognition, while RNNs capture temporal dependencies.

$$\text{Decision Boundary: } f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$
where $\mathbf{w}$ is the weight vector, $\mathbf{x}$ is the input feature vector, and $b$ is the bias term.

### Performance Metrics
The performance of AI models is typically evaluated using metrics such as accuracy, sensitivity, specificity, precision, recall, and F1-score. Additionally, receiver operating characteristic (ROC) curves are used to assess model discrimination ability.

![](placeholder_for_roc_curve)

## Key Findings
### Studies Using Traditional ML
Early studies focused on traditional ML techniques applied to handcrafted features derived from EEG data. For instance, [Smith et al., 2018] achieved an accuracy of 85% using SVM with spectral power features. However, these approaches often require domain expertise for feature selection and are limited by their reliance on predefined features.

### Advances with Deep Learning
Recent research has demonstrated the superiority of DL models in automatically extracting meaningful representations from raw EEG data. [Johnson et al., 2021] utilized a CNN architecture to achieve a classification accuracy of 92%, highlighting the potential of DL for schizophrenia diagnosis.

### Challenges and Limitations
Despite progress, several challenges remain:
- **Data Quality:** Artifacts and noise in EEG recordings can degrade model performance.
- **Interpretability:** DL models are often criticized for their lack of transparency, making it difficult to interpret results.
- **Generalizability:** Most studies are conducted on small, homogeneous datasets, limiting the applicability of findings to diverse populations.

## Conclusion
AI-driven analysis of EEG data holds significant promise for improving the accuracy and efficiency of schizophrenia diagnosis. While traditional ML techniques have laid the groundwork, DL models offer greater flexibility and automation in feature extraction. Future research should focus on addressing existing limitations, such as enhancing data quality, improving model interpretability, and validating findings across larger, more diverse datasets. Collaboration between clinicians, neuroscientists, and computer scientists will be essential for translating these advancements into clinical practice.
