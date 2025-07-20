# 1 Introduction

The convergence of sensing and communication technologies represents a pivotal advancement in the evolution of wireless systems, particularly as we transition from 5G to 6G. This survey explores the integration of sensing and communication (ISC) within the context of 5G-Advanced (5G-A) and 6G networks, highlighting the potential benefits, challenges, and future directions.

## 1.1 Motivation

The demand for higher data rates, lower latency, and enhanced connectivity in 5G-A and 6G is driving the development of new technologies that can support these requirements. Integrated Sensing and Communication (ISC) emerges as a promising solution by leveraging shared resources between communication and sensing functionalities. Traditional approaches have treated these two domains separately, leading to inefficiencies in spectrum utilization and hardware design. ISC aims to address these inefficiencies by enabling dual-use waveforms and shared infrastructure, thereby optimizing resource allocation and improving overall system performance.

## 1.2 Objectives

This survey has several objectives:

1. To provide an overview of the fundamental concepts and principles underlying ISC.
2. To review the key challenges associated with implementing ISC in 5G-A and 6G networks.
3. To analyze existing research on ISC, including early approaches and recent advancements.
4. To discuss architectural frameworks, signal processing techniques, performance metrics, and standardization efforts relevant to ISC.
5. To identify open research issues, future trends, and potential applications of ISC.

## 1.3 Structure of the Survey

The remainder of this survey is organized as follows: Section 2 provides background information on the evolution of communication systems and the fundamentals of ISC, along with key challenges specific to 5G-A and 6G. Section 3 reviews related work, covering early approaches and recent developments in ISC, followed by a comparative analysis of different ISC techniques. Section 4 delves into the main content, discussing architectural frameworks, signal processing techniques, performance metrics, and standardization considerations. Section 5 offers a discussion on open research issues, future trends, and potential applications. Finally, Section 6 concludes the survey with a summary of findings and final remarks.

# 2 Background

## 2.1 Evolution of Communication Systems

The evolution of communication systems has been marked by significant advancements in technology and infrastructure, driven by the increasing demand for higher data rates, lower latency, and enhanced connectivity. The progression from 1G to 5G has seen a paradigm shift from voice-centric services to data-centric services, enabling a wide range of applications such as mobile internet, IoT, and smart cities.

1G systems were primarily analog and focused on voice communication. The transition to 2G introduced digital technologies, which improved spectral efficiency and enabled text messaging. 3G brought about the integration of multimedia services, while 4G further advanced this with high-speed internet access and support for video streaming. 5G, the current generation, aims to provide ultra-reliable low-latency communications (URLLC), massive machine-type communications (mMTC), and enhanced mobile broadband (eMBB).

The next frontier, 6G, is envisioned to extend beyond these capabilities by integrating sensing functionalities into the communication framework, leading to the concept of Integrated Sensing and Communication (ISC). This integration promises to revolutionize various sectors, including autonomous vehicles, environmental monitoring, and healthcare.

## 2.2 Fundamentals of Integrated Sensing and Communication (ISC)

Integrated Sensing and Communication (ISC) refers to the simultaneous use of the same hardware and spectrum for both communication and sensing purposes. Traditional approaches have treated these two functions separately, but ISC leverages the shared resources to achieve more efficient and versatile systems. The core idea is to design waveforms and protocols that can serve dual purposes, thereby reducing redundancy and enhancing overall performance.

### Key Components of ISC

- **Waveform Design**: A critical aspect of ISC is the design of waveforms that can be used for both communication and sensing. These waveforms must balance the requirements of high data rate transmission and accurate target detection. For instance, orthogonal frequency-division multiplexing (OFDM) signals can be modified to include pilot tones for radar-like functionality.

- **Signal Processing**: Advanced signal processing techniques are essential for extracting useful information from the received signals. Techniques such as matched filtering, Doppler estimation, and multi-target tracking are employed to enhance sensing accuracy.

- **Resource Allocation**: Efficient allocation of resources, such as power and bandwidth, is crucial for optimizing the performance of ISC systems. Mathematical models like $\max_{P_s, P_c} \{ R_s + R_c \}$ subject to constraints on power and interference levels can be used to find optimal resource allocations.

### Benefits of ISC

- **Spectral Efficiency**: By sharing the same spectrum, ISC systems can significantly improve spectral efficiency compared to separate communication and sensing systems.

- **Cost Reduction**: Utilizing common hardware reduces the need for additional sensors and transceivers, leading to cost savings.

- **Enhanced Situational Awareness**: Combining communication and sensing provides richer context and better situational awareness, which is particularly beneficial in safety-critical applications.

## 2.3 Key Challenges in ISC for 5G-A and 6G

While ISC offers numerous advantages, it also presents several challenges that need to be addressed to fully realize its potential. These challenges span across technical, regulatory, and practical dimensions.

### Technical Challenges

- **Interference Management**: One of the primary challenges is managing interference between communication and sensing signals. Since both functions share the same spectrum, there is a risk of mutual interference, which can degrade performance. Techniques such as interference cancellation and adaptive filtering are being explored to mitigate this issue.

- **Hardware Complexity**: Implementing ISC requires sophisticated hardware capable of handling multiple tasks simultaneously. This increases complexity and may lead to higher costs and power consumption. Research into reconfigurable antennas and software-defined radios is ongoing to address these concerns.

- **Algorithmic Development**: Developing algorithms that can efficiently process and interpret the combined data from communication and sensing is non-trivial. Machine learning and artificial intelligence are promising avenues for improving the accuracy and reliability of ISC systems.

### Regulatory and Practical Challenges

- **Spectrum Regulation**: Regulatory bodies need to establish guidelines for the coexistence of communication and sensing within the same spectrum. Ensuring fair access and preventing harmful interference will be critical.

- **Standardization**: Standardizing ISC protocols and interfaces is necessary for interoperability and widespread adoption. Collaboration between industry stakeholders and standardization bodies is essential.

- **Policy Implications**: New policies may be required to address privacy and security concerns associated with ISC, especially in scenarios involving sensitive data collection and processing.

In summary, while ISC holds great promise for future communication systems, addressing these challenges will be key to its successful implementation.

# 3 Related Work

The integration of sensing and communication (ISC) has been an evolving field, with early approaches laying the foundation for more sophisticated systems. This section reviews the historical development of ISC, highlights recent advancements, and provides a comparative analysis of various techniques.

## 3.1 Early Approaches to ISC

Early work on ISC primarily focused on coexistence mechanisms where separate systems for communication and radar were integrated into a single platform. The main challenge was to ensure that the two systems did not interfere with each other while achieving their respective objectives. Initial efforts involved time-division multiplexing (TDM) and frequency-division multiplexing (FDM), which allowed radar and communication signals to operate in different time slots or frequency bands, respectively. For instance, TDM-based methods ensured that radar pulses and communication transmissions did not overlap in time, thereby avoiding mutual interference.

$$
t_{\text{radar}} \cap t_{\text{comm}} = \emptyset
$$

However, these methods were limited by inefficiencies in spectrum utilization and hardware complexity. As a result, researchers began exploring more advanced techniques that could exploit the synergies between sensing and communication.

## 3.2 Recent Developments in ISC

Recent advancements in ISC have shifted towards dual-use waveform design, where the same signal can be used for both sensing and communication. This approach leverages the inherent properties of certain waveforms, such as orthogonal frequency-division multiplexing (OFDM), to simultaneously support multiple functionalities. OFDM-based waveforms, for example, can be optimized for both high data rates and accurate target detection.

$$
x(t) = \sum_{k=0}^{N-1} X_k e^{j2\pi k f_0 t}
$$

In addition to waveform design, recent research has also explored joint resource allocation strategies, where the available resources (e.g., power, bandwidth) are dynamically allocated between sensing and communication tasks based on real-time requirements. Machine learning algorithms have been employed to optimize these allocations, leading to improved system performance.

## 3.3 Comparative Analysis of ISC Techniques

A comprehensive comparison of ISC techniques is essential to identify the strengths and limitations of each approach. Table 1 summarizes the key characteristics of different ISC methods, including their spectral efficiency, hardware complexity, and potential applications.

| Technique | Spectral Efficiency | Hardware Complexity | Applications |
| --- | --- | --- | --- |
| Time-Division Multiplexing (TDM) | Low | Low | Coexistence of legacy systems |
| Frequency-Division Multiplexing (FDM) | Medium | Medium | Dual-band operations |
| Dual-Use Waveform Design | High | High | Advanced radar and communication |
| Joint Resource Allocation | High | High | Dynamic environments |

This table highlights the trade-offs involved in selecting an appropriate ISC technique. While TDM and FDM offer simpler implementations, they suffer from lower spectral efficiency. On the other hand, dual-use waveform design and joint resource allocation provide higher performance but require more complex hardware and algorithms.

# 4 Main Content

## 4.1 Architectural Frameworks for ISC

### 4.1.1 Coexistence of Sensing and Communication

The coexistence of sensing and communication in Integrated Sensing and Communication (ISC) systems is a critical aspect that ensures both functionalities operate harmoniously within the same infrastructure. This coexistence can be achieved through various methods, including time-division multiplexing (TDM), frequency-division multiplexing (FDM), and spatial-division multiplexing (SDM). TDM allows sensing and communication to share the same spectrum by alternating their operations over time, while FDM utilizes different frequency bands for each function. SDM leverages multiple antennas to separate sensing and communication signals in space.

In mathematical terms, the coexistence can be described as:

$$
\text{Coexistence} = \begin{cases}
\text{TDM: } & t_s + t_c = T \\
\text{FDM: } & f_s 
eq f_c \\
\text{SDM: } & \mathbf{s}(t) \perp \mathbf{c}(t)
\end{cases}
$$

where $t_s$ and $t_c$ are the durations for sensing and communication, $f_s$ and $f_c$ are the frequencies used for sensing and communication, and $\mathbf{s}(t)$ and $\mathbf{c}(t)$ represent the spatial vectors for sensing and communication signals, respectively.

### 4.1.2 Resource Allocation Strategies

Effective resource allocation is essential for optimizing the performance of ISC systems. The primary resources include spectrum, power, and hardware components. Spectrum allocation involves assigning specific frequency bands to either sensing or communication, ensuring minimal interference between the two functions. Power allocation focuses on distributing available power between sensing and communication tasks to maximize overall system efficiency. Hardware allocation entails utilizing shared or dedicated hardware resources for both sensing and communication.

A common approach to resource allocation is through optimization algorithms that balance the trade-offs between sensing accuracy and communication reliability. For instance, the weighted sum optimization problem can be formulated as:

$$
\max_{P_s, P_c} \alpha R_s(P_s) + (1 - \alpha) R_c(P_c)
$$

where $P_s$ and $P_c$ are the power allocated to sensing and communication, $R_s$ and $R_c$ are the respective performance metrics (e.g., signal-to-noise ratio), and $\alpha$ is a weighting factor that reflects the priority given to sensing versus communication.

### 4.1.3 Hardware Implementations

Hardware implementations for ISC systems must support both sensing and communication functionalities efficiently. Key components include transceivers, antennas, and signal processing units. Transceivers capable of dual-use operation are crucial for enabling simultaneous sensing and communication. Antennas designed for multi-functional use can enhance the spatial separation of signals, reducing interference. Signal processing units need to handle complex waveforms and data fusion tasks to integrate information from both sensing and communication channels.

| Component | Function |
| --- | --- |
| Transceiver | Supports dual-use operation for sensing and communication |
| Antenna | Enhances spatial separation of signals |
| Signal Processing Unit | Handles complex waveforms and data fusion |

## 4.2 Signal Processing Techniques

### 4.2.1 Waveform Design for Dual-Use

Waveform design is a fundamental aspect of ISC systems, aiming to create signals that can serve both sensing and communication purposes effectively. Dual-use waveforms should have desirable properties such as high resolution for sensing and robustness against noise for communication. One approach is to design waveforms with orthogonal frequency division multiplexing (OFDM) structure, which offers flexibility in adapting to different channel conditions.

Mathematically, an OFDM-based dual-use waveform can be represented as:

$$
s(t) = \sum_{k=0}^{N-1} a_k e^{j2\pi f_k t}
$$

where $a_k$ are the complex amplitudes and $f_k$ are the subcarrier frequencies.

### 4.2.2 Interference Management

Interference management is vital for maintaining the integrity of both sensing and communication signals. Techniques such as interference cancellation and adaptive filtering can mitigate unwanted interference. Interference cancellation removes interfering signals from the received signal, while adaptive filtering adjusts filter coefficients dynamically to minimize interference.

An adaptive filter can be modeled as:

$$
y(n) = \sum_{i=0}^{M-1} w_i x(n-i)
$$

where $y(n)$ is the filtered output, $w_i$ are the filter coefficients, and $x(n)$ is the input signal.

### 4.2.3 Data Fusion Methods

Data fusion combines information from multiple sources to improve the accuracy and reliability of ISC systems. Common methods include sensor fusion and decision-level fusion. Sensor fusion integrates raw data from different sensors, while decision-level fusion combines processed decisions from individual sensors.

A typical sensor fusion process can be illustrated as:

![]()

## 4.3 Performance Metrics and Evaluation

### 4.3.1 Sensing Accuracy

Sensing accuracy measures how closely the estimated parameters match the true values. Key metrics include mean squared error (MSE) and root mean squared error (RMSE). These metrics evaluate the precision of sensing results and help identify areas for improvement.

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{x}_i - x_i)^2
$$

where $\hat{x}_i$ and $x_i$ are the estimated and true values, respectively.

### 4.3.2 Communication Reliability

Communication reliability assesses the quality of communication links in ISC systems. Metrics such as bit error rate (BER) and packet loss rate (PLR) are commonly used. A lower BER and PLR indicate more reliable communication.

$$
\text{BER} = \frac{\text{Number of erroneous bits}}{\text{Total number of transmitted bits}}
$$

### 4.3.3 Energy Efficiency

Energy efficiency is crucial for extending the operational lifetime of ISC systems. Metrics like energy consumption per bit ($E_b$) and energy efficiency ($EE$) provide insights into the system's power usage.

$$
EE = \frac{1}{E_b}
$$

## 4.4 Standardization and Regulatory Considerations

### 4.4.1 Spectrum Regulation

Spectrum regulation governs the allocation and usage of frequency bands for ISC systems. Regulatory bodies such as the Federal Communications Commission (FCC) and International Telecommunication Union (ITU) define rules to ensure fair and efficient spectrum utilization.

### 4.4.2 Industry Standards

Industry standards establish guidelines and best practices for implementing ISC systems. Organizations like the Institute of Electrical and Electronics Engineers (IEEE) and 3rd Generation Partnership Project (3GPP) play a pivotal role in developing these standards.

### 4.4.3 Policy Implications

Policy implications encompass the broader societal and economic impacts of ISC technologies. Policies may address issues such as privacy, security, and environmental sustainability, ensuring that ISC systems benefit society while minimizing potential risks.

# 5 Discussion

## 5.1 Open Research Issues

The integration of sensing and communication (ISC) in 5G-A and 6G systems presents numerous open research issues that require further exploration. One of the primary challenges is achieving efficient coexistence between sensing and communication functionalities within a single system. This involves addressing spectrum sharing, interference management, and resource allocation strategies to ensure optimal performance for both functions.

### Spectrum Sharing

Spectrum sharing is critical for ISC as it allows the simultaneous operation of sensing and communication without mutual interference. However, current spectrum allocation policies are often rigid and do not support dynamic sharing. Future research should focus on developing flexible spectrum management techniques that can adapt to varying traffic loads and environmental conditions. Mathematical models such as game theory or optimization algorithms can be employed to find optimal sharing strategies:

$$
\text{Maximize } U = \sum_{i=1}^{N} w_i u_i(\lambda_i)
$$
where $U$ is the overall utility function, $w_i$ are weights representing priorities, and $u_i(\lambda_i)$ are individual utility functions for each user or service.

### Interference Management

Interference management is another significant challenge. The dual-use nature of ISC can lead to increased interference levels, affecting both sensing accuracy and communication reliability. Techniques like advanced signal processing, beamforming, and spatial multiplexing can mitigate interference but need further investigation to enhance their effectiveness in real-world scenarios.

### Resource Allocation Strategies

Effective resource allocation is essential for balancing the demands of sensing and communication. Dynamic resource allocation methods must consider factors such as channel state information (CSI), quality of service (QoS) requirements, and energy efficiency. Machine learning approaches can be explored to predict and adapt resource allocation in real-time.

## 5.2 Future Trends

As ISC continues to evolve, several future trends are expected to shape its development in 5G-A and 6G systems. These trends include advancements in hardware technology, software-defined networking (SDN), artificial intelligence (AI), and new applications.

### Hardware Advancements

Advances in hardware will play a crucial role in enabling more sophisticated ISC systems. For example, the development of millimeter-wave (mmWave) and terahertz (THz) technologies can provide higher bandwidths and better resolution for sensing applications. Additionally, reconfigurable intelligent surfaces (RIS) can enhance signal propagation and reduce interference.

### Software-Defined Networking (SDN)

SDN offers a flexible and programmable approach to network management, which can facilitate the integration of sensing and communication. By decoupling the control plane from the data plane, SDN enables centralized control over network resources, leading to improved resource allocation and interference management.

### Artificial Intelligence (AI)

AI can revolutionize ISC by providing intelligent decision-making capabilities. Machine learning algorithms can optimize waveform design, improve interference mitigation, and enhance data fusion methods. Reinforcement learning, in particular, can be used to adaptively adjust system parameters based on real-time feedback.

### New Applications

Emerging applications such as autonomous vehicles, smart cities, and industrial IoT will drive the demand for advanced ISC systems. These applications require high-precision sensing and reliable communication, pushing the boundaries of what ISC can achieve.

## 5.3 Potential Applications

ISC has the potential to transform various industries and sectors by offering integrated solutions for sensing and communication. Some key applications include:

### Autonomous Vehicles

Autonomous vehicles rely heavily on accurate sensing and reliable communication to navigate safely. ISC can provide real-time environmental awareness through radar and LiDAR sensors while maintaining robust vehicle-to-everything (V2X) communication. This integration ensures seamless coordination between vehicles and infrastructure.

### Smart Cities

Smart cities aim to create efficient and sustainable urban environments. ISC can enable intelligent traffic management, environmental monitoring, and public safety services. For instance, integrated sensing nodes can detect air quality, noise levels, and pedestrian movements, while communication networks disseminate this information to relevant authorities.

### Industrial IoT

In industrial settings, ISC can enhance automation and process control. Sensors embedded in machinery can monitor operational parameters, while communication networks transmit data to central control systems. This integration improves productivity, reduces downtime, and ensures worker safety.

![](placeholder_image_url)

| Application | Sensing Function | Communication Function |
|-------------|------------------|------------------------|
| Autonomous Vehicles | Radar, LiDAR | V2X Communication |
| Smart Cities | Environmental Monitoring | Public Safety Networks |
| Industrial IoT | Operational Monitoring | Control Systems Communication |

# 6 Conclusion
## 6.1 Summary of Findings
This survey has explored the emerging field of Integrated Sensing and Communication (ISC) for 5G-A and 6G systems, highlighting its potential to revolutionize future wireless networks. The evolution from traditional communication systems to integrated frameworks that simultaneously support sensing and communication functionalities has been driven by the need for more efficient spectrum utilization and enhanced situational awareness. Key challenges identified include interference management, resource allocation, and hardware design, which are critical for achieving coexistence between sensing and communication tasks.

The architectural frameworks discussed in this survey provide a foundation for designing ISC systems, emphasizing the importance of coexistence mechanisms, resource allocation strategies, and hardware implementations. Signal processing techniques such as waveform design, interference management, and data fusion methods play a crucial role in optimizing system performance. Performance metrics like sensing accuracy, communication reliability, and energy efficiency have been evaluated to assess the effectiveness of ISC solutions. Additionally, standardization efforts and regulatory considerations, including spectrum regulation and policy implications, are essential for the widespread adoption of ISC technologies.

## 6.2 Final Remarks
In conclusion, ISC represents a promising paradigm shift in the development of next-generation wireless systems. While significant progress has been made, several open research issues remain, particularly in addressing the trade-offs between sensing and communication performance, developing robust interference mitigation techniques, and ensuring compliance with evolving standards and regulations. Future trends suggest that ISC will continue to evolve, driven by advancements in artificial intelligence, machine learning, and new materials. Potential applications of ISC span across various domains, including autonomous vehicles, smart cities, and industrial automation, underscoring its transformative impact on society. As research in this area advances, it is imperative to foster interdisciplinary collaboration to overcome existing challenges and unlock the full potential of ISC for 5G-A and 6G.

