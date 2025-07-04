# 1 Introduction

The integration of sensing and communication (ISAC) represents a transformative paradigm in the evolution of wireless networks, particularly as we transition from 5G to 6G. This survey aims to provide a comprehensive overview of ISAC, exploring its foundational principles, key challenges, recent advancements, and future prospects within the context of next-generation networks.

## 1.1 Motivation

The convergence of sensing and communication functionalities into a unified system is driven by the increasing demand for more efficient, reliable, and versatile wireless networks. Traditional approaches have treated sensing and communication as separate entities, leading to inefficiencies in spectrum utilization and hardware design. ISAC seeks to address these limitations by leveraging shared resources, such as frequency bands and antennas, thereby enhancing both the performance and flexibility of wireless systems. Moreover, the integration of sensing capabilities can significantly improve situational awareness, enabling advanced applications like autonomous driving, smart cities, and public safety.

## 1.2 Objectives

This survey has several objectives:

- To provide an in-depth understanding of the fundamental concepts and challenges associated with ISAC.
- To review the evolution of 5G and 6G networks and highlight how ISAC fits into this progression.
- To examine early and recent approaches to ISAC, including their strengths and limitations.
- To explore various architectures, signal processing techniques, and applications of ISAC.
- To discuss performance metrics, trade-offs, and optimization strategies in ISAC systems.
- To identify current limitations, open issues, and potential research directions.
- To analyze the implications of ISAC on standardization and policy-making.

## 1.3 Structure of the Survey

The remainder of this survey is organized as follows: Section 2 provides background information on the evolution of 5G and 6G networks and introduces the fundamentals of ISAC, along with key challenges. Section 3 reviews related work, covering early and recent developments in ISAC. Section 4 delves into the main content, discussing architectures, signal processing techniques, applications, and performance evaluation. Section 5 offers a discussion on current limitations, future research directions, and standardization efforts. Finally, Section 6 concludes the survey with a summary of findings and final remarks.

# 2 Background

The rapid advancement of wireless communication technologies has led to the emergence of 5G and the anticipation of 6G networks. This section provides a comprehensive background on the evolution of these networks, introduces the fundamentals of Integrated Sensing and Communication (ISAC), and outlines the key challenges associated with ISAC.

## 2.1 Evolution of 5G and 6G Networks

The transition from 4G to 5G marked a significant leap in network capabilities, characterized by higher data rates, lower latency, and massive connectivity. 5G networks leverage advanced technologies such as millimeter-wave (mmWave) frequencies, massive MIMO (Multiple-Input Multiple-Output), and network slicing to meet the demands of diverse applications like augmented reality (AR), virtual reality (VR), and Internet of Things (IoT). The evolution towards 6G is driven by the need for even greater performance improvements, including terahertz (THz) frequency bands, ultra-reliable low-latency communications (URLLC), and the integration of artificial intelligence (AI) and machine learning (ML).

![](placeholder_for_5g_6g_evolution_diagram)

## 2.2 Fundamentals of Integrated Sensing and Communication (ISAC)

Integrated Sensing and Communication (ISAC) aims to combine the functionalities of sensing and communication within a single system. Traditional systems separate these functions, leading to inefficiencies in resource utilization and hardware redundancy. ISAC leverages shared resources, such as antennas, signal processing units, and spectrum, to achieve both objectives simultaneously. The core principle of ISAC lies in the joint design of sensing and communication signals, enabling mutual benefits such as improved spectral efficiency and enhanced situational awareness.

Mathematically, the ISAC system can be represented as:
$$
y(t) = h_s(t) \cdot x_s(t) + h_c(t) \cdot x_c(t) + n(t)
$$
where $y(t)$ is the received signal, $h_s(t)$ and $h_c(t)$ are the channel responses for sensing and communication, respectively, $x_s(t)$ and $x_c(t)$ are the transmitted sensing and communication signals, and $n(t)$ is the noise.

## 2.3 Key Challenges in ISAC

Several challenges must be addressed to realize the full potential of ISAC systems. One major challenge is the coexistence of sensing and communication signals without mutual interference. Ensuring that the two types of signals do not degrade each other's performance requires sophisticated signal design and interference management techniques. Another challenge is the development of efficient hardware architectures that can support both functions while maintaining cost-effectiveness and power efficiency. Additionally, there is a need for new performance metrics that can evaluate the trade-offs between sensing accuracy and communication reliability.

| Challenge | Description |
| --- | --- |
| Coexistence of Signals | Preventing interference between sensing and communication signals |
| Hardware Design | Developing cost-effective and power-efficient hardware |
| Performance Metrics | Establishing metrics for evaluating ISAC systems |

# 3 Related Work

The related work section provides a comprehensive overview of the historical development and recent advancements in Integrated Sensing and Communication (ISAC) systems. This section aims to highlight the evolution of ISAC, from its early approaches to the latest developments tailored for 5G and 6G networks, and concludes with a comparative analysis of various ISAC techniques.

## 3.1 Early Approaches to ISAC

Early research on ISAC primarily focused on integrating radar and communication functionalities within a single system. The initial efforts were driven by the need to optimize spectrum usage and enhance operational efficiency. One of the pioneering works in this area was conducted by [Author et al., Year], who proposed a co-design framework that leveraged shared hardware resources between radar and communication modules. This approach significantly reduced the overall system complexity and cost while improving spectral efficiency.

Mathematically, the joint design can be formulated as:
$$
\min_{x} f(x) \quad \text{subject to} \quad g(x) \leq 0,
$$
where $f(x)$ represents the objective function that balances performance metrics such as sensing accuracy and communication throughput, and $g(x)$ denotes the constraints imposed by hardware limitations.

![]()

## 3.2 Recent Developments in ISAC for 5G/6G

Recent advancements in ISAC have been largely influenced by the rapid development of 5G and the anticipation of 6G networks. These networks demand higher data rates, lower latency, and enhanced reliability, which necessitate innovative solutions for integrated sensing and communication. A notable trend is the integration of advanced signal processing techniques, such as beamforming and massive MIMO, to improve both sensing and communication capabilities.

For instance, [Author et al., Year] introduced a novel architecture that combines millimeter-wave (mmWave) communications with high-resolution radar sensing. This architecture exploits the rich scattering environment at mmWave frequencies to achieve superior spatial resolution and data throughput. Additionally, the use of reconfigurable intelligent surfaces (RIS) has emerged as a promising technology to enhance ISAC performance in non-line-of-sight (NLOS) scenarios.

| Feature | 5G ISAC | 6G ISAC |
|---------|---------|---------|
| Frequency Band | Sub-6 GHz, mmWave | THz |
| Data Rate | Gbps | Tbps |
| Latency | ms | μs |

## 3.3 Comparative Analysis of ISAC Techniques

This subsection compares different ISAC techniques based on key performance indicators such as spectral efficiency, energy consumption, and system complexity. Various studies have explored different methodologies, including time-sharing, frequency-sharing, and joint design approaches. Each technique has its own advantages and trade-offs, which are summarized in Table 1.

| Technique | Spectral Efficiency | Energy Consumption | System Complexity |
|-----------|---------------------|--------------------|-------------------|
| Time-Sharing | Moderate | Low | Low |
| Frequency-Sharing | High | Medium | Medium |
| Joint Design | Very High | High | High |

In conclusion, the choice of ISAC technique depends on the specific application requirements and available resources. For example, time-sharing is suitable for applications where low energy consumption is critical, whereas joint design approaches offer the best performance but come at a higher cost and complexity.

# 4 Main Content

## 4.1 Architectures for ISAC Systems

The architecture of Integrated Sensing and Communication (ISAC) systems is crucial for enabling the seamless integration of sensing and communication functionalities. This section explores various architectural designs that facilitate coexistence, joint design approaches, and hardware implementations.

### 4.1.1 Coexistence of Sensing and Communication

Coexistence refers to the ability of an ISAC system to perform both sensing and communication tasks simultaneously without significant interference. One approach is time-division multiplexing (TDM), where the system alternates between sensing and communication in predefined time slots. Another method is frequency-division multiplexing (FDM), which allocates different frequency bands for sensing and communication. Mathematically, this can be represented as:

$$
f_{sensing} 
eq f_{communication}
$$

where $f_{sensing}$ and $f_{communication}$ denote the frequencies used for sensing and communication, respectively. ![](placeholder_for_coexistence_diagram)

### 4.1.2 Joint Design Approaches

Joint design approaches aim to optimize both sensing and communication performance by integrating their functionalities at a deeper level. This includes joint signal processing, resource allocation, and waveform design. For instance, a joint radar-communication (JRC) system can use the same waveform for both purposes, reducing hardware complexity and improving spectral efficiency. The key challenge here is balancing the trade-offs between sensing accuracy and communication reliability.

### 4.1.3 Hardware Implementations

Hardware implementations for ISAC systems require advanced technologies such as millimeter-wave (mmWave) and massive multiple-input multiple-output (MIMO) antennas. These technologies enable high-resolution sensing and high-speed communication. A typical ISAC hardware setup might include a shared frontend with separate backends for processing sensing and communication data. | Component | Function |
| --- | --- |
| Shared Frontend | Handles RF signals for both sensing and communication |
| Sensing Backend | Processes radar signals for target detection |
| Communication Backend | Manages communication protocols and data transmission |

## 4.2 Signal Processing Techniques

Signal processing plays a vital role in enhancing the performance of ISAC systems. This section discusses three key techniques: sensing signal design, interference management, and data fusion methods.

### 4.2.1 Sensing Signal Design

Designing effective sensing signals is essential for achieving high-resolution and accurate sensing. Waveforms such as linear frequency modulation (LFM) and stepped-frequency continuous wave (SFCW) are commonly used due to their favorable properties. LFM signals provide good range resolution, while SFCW signals offer better Doppler resolution. The choice of waveform depends on the specific application requirements.

### 4.2.2 Interference Management

Interference management is critical in ISAC systems to ensure reliable communication in the presence of sensing activities. Techniques like adaptive filtering and beamforming can mitigate interference. Adaptive filtering adjusts filter coefficients based on the interference environment, while beamforming focuses the transmitted energy in specific directions to avoid interference. Mathematically, the interference suppression can be modeled as:

$$
y(t) = x(t) - \sum_{i=1}^{N} h_i(t) * i(t)
$$

where $y(t)$ is the output signal, $x(t)$ is the desired signal, $h_i(t)$ represents the channel response, and $i(t)$ denotes the interference.

### 4.2.3 Data Fusion Methods

Data fusion combines information from multiple sensors and communication channels to improve overall system performance. Common fusion methods include Kalman filtering and particle filtering. Kalman filtering provides optimal estimates for linear systems, while particle filtering handles non-linear and non-Gaussian scenarios. The effectiveness of data fusion depends on the quality and diversity of input data.

## 4.3 Applications of ISAC

ISAC systems have a wide range of applications across various domains. This section highlights three prominent areas: automotive radar and V2X communications, smart cities and IoT, and public safety and surveillance.

### 4.3.1 Automotive Radar and V2X Communications

In the automotive industry, ISAC systems enhance vehicle-to-everything (V2X) communications by integrating radar sensing for obstacle detection and communication for data exchange. This improves road safety and traffic efficiency. The integration allows vehicles to perceive their surroundings and communicate with other vehicles and infrastructure simultaneously.

### 4.3.2 Smart Cities and IoT

Smart cities leverage ISAC systems to monitor environmental conditions, manage traffic flow, and optimize resource usage. IoT devices equipped with ISAC capabilities can gather real-time data and communicate it to central systems for analysis. This enables more intelligent and responsive urban management.

### 4.3.3 Public Safety and Surveillance

Public safety applications benefit from ISAC systems through enhanced situational awareness. Surveillance systems can detect and track objects while communicating relevant information to authorities. This improves emergency response times and enhances security measures.

## 4.4 Performance Metrics and Evaluation

Evaluating the performance of ISAC systems requires a comprehensive set of metrics that cover both sensing and communication aspects. This section outlines key metrics and discusses trade-offs and optimization strategies.

### 4.4.1 Sensing Accuracy and Resolution

Sensing accuracy refers to how closely the measured values match the true values, while resolution indicates the smallest detectable change. High accuracy and resolution are critical for applications like automotive radar. Metrics such as mean squared error (MSE) and signal-to-noise ratio (SNR) are commonly used to assess sensing performance.

### 4.4.2 Communication Reliability and Efficiency

Communication reliability ensures that data is transmitted accurately and efficiently. Key metrics include bit error rate (BER) and throughput. Efficient resource allocation and robust error correction schemes are essential for maintaining high reliability.

### 4.4.3 Trade-offs and Optimization

Optimizing ISAC systems involves balancing trade-offs between sensing accuracy and communication reliability. For example, increasing the sensing bandwidth may degrade communication performance. Optimization techniques such as multi-objective optimization and machine learning can help find the best compromise. Mathematical models can represent these trade-offs as:

$$
\min f(x) = w_1 \cdot g_1(x) + w_2 \cdot g_2(x)
$$

where $f(x)$ is the objective function, $g_1(x)$ and $g_2(x)$ represent the sensing and communication performance metrics, and $w_1$ and $w_2$ are weighting factors.

# 5 Discussion

## 5.1 Current Limitations and Open Issues

The integration of sensing and communication (ISAC) in 5G and 6G networks presents several challenges that must be addressed to fully realize its potential. One major limitation is the coexistence of sensing and communication functionalities within the same system. The design of ISAC systems requires careful consideration of interference between the two functions, which can degrade performance. For instance, the radar signals used for sensing may interfere with communication signals, leading to reduced data rates or increased error rates.

Another challenge lies in achieving high-resolution sensing while maintaining reliable communication. Sensing accuracy depends on factors such as signal-to-noise ratio (SNR), bandwidth, and resolution, which are often at odds with the requirements for efficient communication. This trade-off necessitates the development of advanced signal processing techniques that can optimize both sensing and communication performance.

Moreover, hardware implementation poses significant challenges. Existing hardware architectures are primarily designed for either communication or sensing, making it difficult to integrate both functionalities without compromising performance. Developing hardware that supports ISAC efficiently will require innovations in antenna design, RF front-ends, and signal processing units.

### Open Issues

Several open issues remain unresolved in the field of ISAC. One critical issue is the lack of a unified framework for evaluating ISAC systems. Performance metrics for ISAC systems should encompass both sensing and communication aspects, but current evaluation methods often focus on one aspect at the expense of the other. Establishing comprehensive performance metrics that account for both sensing accuracy and communication reliability is essential for fair comparisons and meaningful progress.

Additionally, there is a need for more research on dynamic resource allocation strategies. ISAC systems must adapt to varying environmental conditions and user demands, which calls for intelligent algorithms that can dynamically allocate resources such as power, bandwidth, and time slots. These algorithms should also consider the trade-offs between sensing and communication performance to ensure optimal overall system performance.

## 5.2 Future Research Directions

Future research in ISAC should focus on addressing the limitations and open issues discussed above. One promising direction is the development of joint design approaches that integrate sensing and communication from the ground up. Joint design can lead to more efficient use of resources and better performance compared to separate designs. For example, joint waveform design can exploit the commonalities between radar and communication waveforms to achieve mutual benefits.

Another important area of research is the exploration of new technologies that can enhance ISAC performance. Millimeter-wave (mmWave) and terahertz (THz) frequencies offer wider bandwidths and higher resolutions, making them ideal candidates for ISAC applications. However, these frequencies also introduce new challenges, such as increased path loss and sensitivity to environmental factors. Research into robust mmWave and THz ISAC systems is necessary to overcome these challenges.

Furthermore, machine learning (ML) and artificial intelligence (AI) have the potential to revolutionize ISAC by enabling intelligent decision-making and adaptive resource management. ML algorithms can learn from historical data to predict future conditions and optimize system parameters in real-time. For instance, reinforcement learning can be used to develop policies for dynamic resource allocation that maximize both sensing and communication performance.

Finally, the integration of ISAC with emerging network paradigms such as edge computing and network slicing can unlock new possibilities. Edge computing can provide low-latency processing for ISAC applications, while network slicing can create dedicated virtual networks for different ISAC services. Investigating the synergies between ISAC and these paradigms can lead to innovative solutions that enhance the capabilities of 5G and 6G networks.

## 5.3 Standardization and Policy Implications

Standardization plays a crucial role in the deployment and adoption of ISAC technologies. Currently, there is no widely accepted standard for ISAC systems, which hinders interoperability and large-scale deployment. Developing a standardized framework for ISAC is essential to ensure seamless integration with existing 5G and 6G infrastructure. This framework should define key components such as waveform design, signal processing algorithms, and hardware specifications.

Policy implications also need to be considered. Regulatory bodies must address the unique challenges posed by ISAC, such as spectrum management and privacy concerns. Spectrum allocation policies should accommodate the dual-use nature of ISAC systems, ensuring that both sensing and communication functionalities have access to sufficient spectrum. Privacy concerns arise from the potential misuse of sensing data, particularly in applications like public safety and surveillance. Policies should establish guidelines for data collection, storage, and sharing to protect individual privacy.

In conclusion, the successful deployment of ISAC in 5G and 6G networks depends on addressing current limitations, exploring future research directions, and establishing appropriate standards and policies. Collaborative efforts among researchers, industry stakeholders, and policymakers are necessary to overcome the challenges and unlock the full potential of ISAC.

# 6 Conclusion

## 6.1 Summary of Findings

This survey has provided a comprehensive overview of Integrated Sensing and Communication (ISAC) systems for 5G and 6G networks. The evolution from traditional communication systems to the integration of sensing capabilities marks a significant advancement in wireless technology. Key findings include:

- **Architectural Innovations**: ISAC systems leverage coexistence and joint design approaches, enabling simultaneous operation of sensing and communication functionalities. This coexistence is crucial for achieving high efficiency and performance.
- **Signal Processing Techniques**: Advanced signal processing methods, such as interference management and data fusion, are essential for enhancing both sensing accuracy and communication reliability.
- **Applications**: ISAC finds applications in diverse fields, including automotive radar and V2X communications, smart cities, IoT, and public safety, demonstrating its versatility and potential impact.
- **Performance Metrics**: Evaluating ISAC systems requires a balance between sensing accuracy ($\text{SAR}$) and communication reliability ($\text{BER}$), with optimization techniques addressing trade-offs.

## 6.2 Impact on 5G/6G Development

The integration of sensing and communication in 5G and 6G networks will revolutionize how these technologies are deployed and utilized. ISAC offers several advantages that can significantly influence the development of future wireless systems:

- **Enhanced Spectrum Utilization**: By combining sensing and communication, ISAC maximizes spectrum efficiency, a critical resource in crowded frequency bands.
- **Improved Situational Awareness**: Real-time environmental monitoring through ISAC enhances situational awareness, benefiting autonomous vehicles, drones, and public safety applications.
- **Cost and Energy Efficiency**: Joint infrastructure for sensing and communication reduces hardware costs and energy consumption, promoting sustainable network deployment.

| Impact Area | Description |
| --- | --- |
| Spectrum Utilization | Efficient use of limited spectrum resources |
| Situational Awareness | Enhanced real-time monitoring and decision-making |
| Cost and Energy Efficiency | Reduced infrastructure and operational costs |

## 6.3 Final Remarks

In conclusion, ISAC represents a transformative approach to integrating sensing and communication in next-generation wireless networks. While challenges remain, particularly in standardization and policy implications, the potential benefits are substantial. Future research should focus on overcoming current limitations, exploring new architectures, and developing robust evaluation frameworks. As 5G evolves into 6G, ISAC will play a pivotal role in shaping the future of wireless communications, offering unprecedented capabilities and applications.

