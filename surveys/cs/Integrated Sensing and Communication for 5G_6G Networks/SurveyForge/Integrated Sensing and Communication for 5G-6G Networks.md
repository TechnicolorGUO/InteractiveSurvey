# Integrated Sensing and Communication for 5G-6G Networks: A Comprehensive Survey

## 1 Introduction

Integrated Sensing and Communication (ISAC) serves as a cornerstone in the evolution of 5G and 6G networks, synthesizing the dual functions of communication and sensing within a unified framework. Through this integration, networks can achieve superior spectrum efficiency, reduced hardware costs, and enhanced system capabilities, addressing the pressing demands of modern applications including autonomous systems, smart cities, and healthcare [1; 2]. The historical context of this integration traces back to the separate but parallel advancements in communication and sensing technologies, with recent convergence driven by the increasing demand for resource-efficient and high-capability networks [3; 4].

The concept of ISAC is fundamentally linked to the shift from traditional resource allocation strategies to more dynamic and integrated approaches. Historically, communication networks focused primarily on maximizing data throughput while separate systems dedicated to sensing evolved for radar and surveillance applications. The recognition that these dual functionalities could share spectral and hardware resources marks a significant paradigm shift. The joint systems enable not only efficient spectrum utilization but also the capability to support emerging applications that require both high-resolution sensing and high-speed communication [5; 6].

Defining ISAC within the realm of 5G and 6G entails understanding its scope, which includes leveraging cutting-edge technologies such as massive multiple-input multiple-output (MIMO) and reconfigurable intelligent surfaces (RIS). These technologies not only enhance communication links but also improve the spatial resolution and accuracy of sensing tasks [7; 8]. The dual-use of resources introduces complexities, particularly in interference management, wherein non-orthogonal multiple access (NOMA) and sophisticated signal processing techniques emerge as key solutions for mitigating intra-system interference and optimizing resource allocation [9; 10].

The implications of ISAC extend beyond technological advancements. The joint framework enhances system capability, allowing for seamless support of novel applications such as intelligent transportation systems and real-time environmental monitoring. However, this integration also brings forth a series of challenges, including privacy concerns and the need for robust standardization to ensure system interoperability across diverse platforms and environments [11; 12]. Addressing these challenges is essential to realize the full potential of ISAC in future networks.

Emerging trends in ISAC indicate a movement towards distributed and intelligent systems, paving the way for more resilient and adaptable network architectures. Employing machine learning and artificial intelligence for signal processing and resource management underpins these advancements, offering solutions that dynamically adapt to network conditions and user demands [3; 13]. As research progresses, the ISAC paradigm is expected to transform the foundational principles of communication systems, facilitating a more integrated and responsive technological ecosystem.

In conclusion, the integration of sensing and communication reflects a transformative approach in networking, where the boundaries between communication and sensing blur to form a comprehensive and versatile system. This integration not only meets the current demands for efficiency and functionality but also lays the groundwork for future network innovations that capitalize on the synergistic capabilities of ISAC.

## 2 Enabling Technologies and Architectures

### 2.1 High-Frequency Communication Technologies

High-frequency communication technologies, namely millimeter-wave (mmWave) and terahertz (THz) communications, have emerged as pivotal enablers of integrated sensing and communication systems in the evolution from 5G to 6G networks. These technologies are particularly significant due to their potential to provide expansive bandwidths and facilitate the fusion of communication and sensing functions [6]. As the demand for higher data rates and precision sensing in next-generation networks escalates, understanding the roles, challenges, and innovations associated with mmWave and THz communications becomes imperative.

Millimeter-wave communication operates roughly between 30 to 300 GHz and is pivotal for 5G and beyond due to its capacity to offer high data rates and large bandwidth [14]. The propagation characteristics of mmWaves present unique advantages, including high spatial resolution and beamforming capabilities which are essential for integrated sensing tasks [6]. However, these high frequencies are prone to significant attenuation due to atmospheric absorption and require line-of-sight (LOS) pathways, posing challenges in non-LOS urban environments [15].

To mitigate these challenges, advanced antenna technologies such as massive MIMO (Multiple-Input Multiple-Output) systems are employed to enhance beamforming and spatial diversity, thus improving signal reliability [16]. Additionally, protocols for adaptive beam steering and dynamic spectrum allocation are being developed to optimize mmWave usage and spectrum efficiency in dense urban scenarios [5].

Terahertz communication, spanning frequencies from 100 GHz to 10 THz, offers even more extensive bandwidth and is anticipated to facilitate ultra-high-speed communication and high-resolution sensing in 6G networks [17]. The high frequency range enables applications requiring detailed environmental sensing and cm-level localization accuracy [17]. However, THz waves suffer even greater from atmospheric attenuation and molecular absorption, necessitating innovations in channel modeling and materials to enhance transmission efficiency [3].

Emerging trends in THz technology involve the exploration of new materials and device architectures that enhance the physical layer's performance in handling THz signals. The integration of artificial intelligence for channel prediction and resource allocation further aids in overcoming propagation challenges [13]. Moreover, collaborative advances such as machine learning-driven adaptive coding and modulation schemes present credible solutions to mitigate effects of path loss and signal distortion [14].

In conclusion, the continued exploration and integration of mmWave and THz communications are vital for achieving the dual goals of high-capacity data transmission and precise environmental sensing in 6G networks. Future research directions must focus on addressing inherent propagation issues and enhancing the interoperability of high-frequency systems with existing communication infrastructures. Innovations in antenna design, dynamic spectrum management, and smart material usage will be critical in actualizing the full potential of these high-frequency communication technologies, underscoring their role as cornerstones in the realization of integrated sensing and communication platforms. The pursuit of such advancements will not only advance telecommunications but also catalyze the development of intelligent, adaptable networks capable of responding to the diverse needs of future wireless applications.

### 2.2 Advanced Antenna and Surface Technologies

In the rapidly evolving landscape of Integrated Sensing and Communication (ISAC) for 5G-6G networks, advanced antenna and surface technologies, notably Massive Multiple-Input Multiple-Output (MIMO) systems and Reconfigurable Intelligent Surfaces (RIS), are pivotal. These technologies significantly enhance communication capacities and sensing capabilities, enabling the optimization of both spatial and spectral efficiency crucial for these advanced networks.

Massive MIMO technologies are crucial due to their capacity to utilize large antenna arrays, thereby substantially increasing channel capacity and spatial resolution, which are essential for integrated sensing tasks. The shift from conventional to massive MIMO systems involves managing an extensive number of antennas, which heightens computational complexity and power demands. However, leveraging techniques such as advanced beamforming and spatial multiplexing allows massive MIMO systems to direct energy precisely toward desired directions, enhancing both communication throughput and sensing precision [18]. Notably, these systems exploit spatial degrees of freedom to separate and extract multiple signals from co-located objects, profoundly benefiting the integration of communication and sensing [19].

Alongside MIMO technologies, Reconfigurable Intelligent Surfaces (RIS) bring a unique dimension by dynamically altering the wireless medium itself. By manipulating electromagnetic properties, RIS can beneficially modify wireless environments, improving signal propagation, extending coverage, and ameliorating received signal quality by mitigating multipath fading and interference [20]. RIS integration amplifies network adaptability through dynamic wavefront control, enabling precise signal delivery to intended targets, maximizing spectrum efficiency, and supporting high-precision localization and environmental sensing [20].

Further enriching these capabilities is the innovation of Simultaneously Transmitting and Reflecting (STAR) surfaces, which allow surfaces to transmit and reflect signals simultaneously, offering new flexibility in controlling signal paths. STAR surfaces effectively double available communication paths in a network, enhancing coverage while optimizing resource allocation for communication and sensing functionalities [21].

These technologies come with notable trade-offs and challenges. Massive MIMO systems encounter challenges related to signal processing complexity and energy requirements, particularly in ultra-dense networks. On the other hand, RIS technologies require sophisticated control algorithms for optimal performance, especially in dynamic environments with unpredictable fading and interference [22]. Integration efforts must address practical issues like effective calibration across large antenna arrays, real-time adaptive beamforming, and developing scalable algorithms for real-time environmental sensing and responsiveness [13].

Emerging trends point toward a collaborative deployment of MIMO and RIS technologies to leverage their combined strengths, potentially enabling ISAC systems in 6G networks to achieve exceptional efficiency and precision. Ongoing research focusing on optimizing these technologies through machine learning and AI offers promising directions for overcoming technical constraints and enhancing network performance [5].

In summary, advancements in antenna and surface technologies significantly enhance current ISAC systems and lay the groundwork for transformative growth in next-generation networks. The interplay between MIMO and RIS, along with innovations like STAR surfaces, heralds significant progress in addressing the multifaceted demands of future communication and sensing networks. As progress continues, these technologies are expected to become increasingly integrated and sophisticated, unlocking new potentials for seamless spectrum sharing and environment-aware communication systems.

### 2.3 Architectural Frameworks for Integration

The architectural frameworks for integrating sensing and communication (ISAC) functionalities in advanced networks are pivotal to the realization of efficient and seamless interoperability. The integration of these dual capabilities aims to optimize resource utilization, latency, and overall operational efficiency in future 5G and 6G networks. As such, several architectural frameworks have been proposed, each with unique features and trade-offs.

One core architectural approach is the distributed intelligent architecture, which leverages edge computing and distributed network functions to enhance scalability and flexibility in ISAC systems. This architecture decentralizes processing tasks, thus reducing latency and improving local data processing, a necessary feature for applications requiring real-time data analytics [23]. Distributed intelligence also facilitates the dynamic adjustment of network resources based on sensing and communication demands, providing a more responsive and adaptable network environment [24].

Semantic and goal-oriented communication frameworks represent another innovative approach, emphasizing the alignment of data transmission with user-specific goals. This method diverges from traditional data-centric models, employing semantic analysis to prioritize information transfer that directly impacts user objectives [22]. Although this model promises more efficient communication by reducing redundant data transmissions, it poses significant challenges in semantic information processing and context understanding.

The integration of cloud-computing resources with communication network functions forms another architectural blueprint, enhancing the processing capability of ISAC systems. Cloud-network integration enables the offloading of complex computations to centralized cloud resources, allowing for more sophisticated data processing and storage solutions. This framework promotes a high degree of centralized control and coordination across widespread network elements, supporting complex tasks such as large-scale data analysis and machine learning applications [25]. However, the latency introduced by cloud-based processing is a critical challenge that must be addressed through optimized communication protocols and infrastructure improvements.

Evaluating the strengths and limitations of these frameworks exposes several trade-offs. Distributed architectures offer reduced latency and immediate data processing, yet may suffer from a lack of cohesive system-wide coordination. Conversely, cloud-based frameworks provide robust processing capabilities and centralized control but introduce latency issues that could hinder real-time operations. Semantic frameworks hold the potential for communication efficiency but require advanced cognitive and processing capabilities to interpret semantic content precisely.

Notable emerging trends that influence architectural development in ISAC systems include the growing reliance on machine learning and artificial intelligence to enhance system adaptability and the integration of reconfigurable intelligent surfaces (RIS) to dynamically manipulate signal propagation environments. The implementation of RIS technologies could potentially transform wireless communication landscapes by offering programmable radio environments, enhancing coverage, and improving energy and spectrum efficiencies [23; 26].

In conclusion, the evolution of architectural frameworks for ISAC represents a multi-dimensional challenge encompassing technological innovation, computational efficiency, and practical deployment considerations. Future research should focus on harmonizing these diverse architectural approaches to create a cohesive ecosystem that maximizes the potential of integrated sensing and communication functionalities. Efforts should also be directed towards addressing existing limitations in latency, interoperability, and semantic processing, facilitating a seamless transition into the next generation of network technologies.

### 2.4 Signal Design and Processing Techniques

Signal design and processing techniques are crucial for optimizing the performance of Integrated Sensing and Communication (ISAC) systems, especially in the advanced architectures of 5G and 6G networks. These innovations are designed to manage interference, enhance efficiency, and improve coherence, thereby ensuring seamless integration and effective utilization of wireless resources. 

A primary focus in this domain is joint waveform design, aiming to develop a singular waveform capable of supporting both communication and sensing functions. The challenge lies in crafting waveforms that maximize mutual information between transmission and reception while remaining resilient against interference common in shared spectrum environments. Orthogonal Time Frequency Space (OTFS) modulation emerges as a promising candidate, offering robustness against multipath delay-Doppler shifts and maintaining coherence across temporal and frequency domains [27].

Interference management is critical due to the intricate task of managing simultaneous sensing and communication. Techniques such as non-orthogonal multiple access (NOMA) show significant promise within ISAC frameworks by efficiently coordinating resources for diverse tasks, thereby mitigating inter-user and sensing-to-communication interference prevalent in joint systems [28].

The complexity of designing effective ISAC systems necessitates advanced signal optimization algorithms. Cutting-edge methodologies, underpinned by machine learning and advanced statistical models, refine signal processing to enhance power efficiency and reduce latency. Deep learning-based strategies have gained prominence, empowering real-time adaptation and optimization in dynamic environments [29].

Simultaneously, the integration of intelligent reflecting surfaces (IRS) opens new possibilities for ISAC systems by dynamically altering wireless propagation paths. This IRS technology facilitates superior spatial signal management, enhancing both sensing capabilities and communication coverage [30]. However, deploying these surfaces introduces unique challenges in coherent phase alignment and precise channel estimation, which require further research [31].

Emerging trends highlight distributed architectures that promote scalable and energy-efficient ISAC operations. Cooperative sensing and communication paradigms are particularly enticing, offering distributed network intelligence through joint spatial and spectral resource allocation, significantly enhancing network-wide performance metrics [32].

Moving forward, research must focus on refining semantic frameworks that prioritize meaningful data transmission over classical throughput maximization, aligning with ongoing architectural advancements. This paradigm shift could redefine performance metrics, emphasizing data significance and goal-oriented communication to ensure optimal resource allocation [33].

In conclusion, the success of these signal design and processing interventions depends on their ability to balance the dual demands of communication and sensing, innovate past traditional frequency limitations, and adapt to diverse and dynamic environments. This pursuit aligns naturally with computational advancements discussed in subsequent subsections, advancing towards fully integrated and efficient ISAC systems in next-generation wireless communication research.

### 2.5 Computational and AI-Driven Enhancements

The rapidly evolving landscape of Integrated Sensing and Communication (ISAC) systems within 5G and 6G networks increasingly leverages computational advancements, particularly artificial intelligence (AI), to enhance system capabilities. This subsection examines the pivotal role of computational techniques and AI in advancing ISAC systems, detailing both existing methodologies and potential future developments.

At the core of computational enhancements in ISAC is the integration of machine learning (ML) techniques, which are pivotal in optimizing signal processing, channel estimation, and resource allocation. Leveraging ML, particularly deep learning (DL), allows systems to predict channel states, adapt beamforming strategies, and manage multi-user interference more effectively. For instance, AI-driven methods can dynamically allocate resources based on predicted traffic patterns, optimizing both spectrum use and service quality by adapting to changing network conditions and user demands [34].

Edge computing also plays a critical role in reducing latency and improving responsiveness in ISAC systems. By processing data closer to the source, edge computing minimizes the time required to transmit data to central clouds and back. This localized processing is particularly beneficial for applications requiring low-latency responses, such as autonomous vehicles or real-time health monitoring systems. The integration of edge computing with existing communication infrastructure advances the seamless operation of ISAC applications by ensuring real-time data processing and immediate action [35].

Another revolutionary development is AI-enabled channel modeling, which enhances the accuracy of channel state information (CSI) through predictive analytics. Advanced machine learning models can infer CSI from limited measurements, thereby facilitating reliable communication in dynamic and complex environments. These models support the dynamic adaptation and optimization of ISAC functionalities, ensuring robust performance in the presence of multipath fading and mobility [34].

The trade-off between communication efficiency and sensing accuracy is a persistent challenge in ISAC systems. AI-based algorithms provide novel approaches to tackle these trade-offs by optimizing sensing and communication objectives jointly. For example, resource management techniques can employ AI to allocate spectrum and power resources, effectively balancing sensing and communication needs while considering the impact of user density and environmental factors [36].

Despite the notable advancements, challenges remain in fully realizing the potential of computational and AI-driven enhancements in ISAC systems. The integration of diverse AI models into existing architectures demands significant computational resources, raising energy consumption concerns—a critical issue in 5G/6G networks that aim for sustainability and reduced carbon footprints [37]. Additionally, the reliability of AI models is contingent on the quality and quantity of training data, which can be scarce in emergent or less-studied use-case scenarios.

Looking forward, the future of computational and AI-driven ISAC advancements involves concerted research efforts in developing lightweight AI models that maintain high precision with reduced computational overhead. Continued exploration into federated learning paradigms presents an opportunity for improving model training across distributed networks without compromising data privacy or incurring excessive communication costs [15]. Moreover, the convergence of AI with emerging technologies, such as quantum computing, could offer unprecedented capabilities and efficiencies in ISAC systems, profoundly impacting both theoretical advancements and practical applications.

In conclusion, computational and AI-driven enhancements are set to remain at the forefront of ISAC development, providing the necessary tools to overcome current limitations and drive innovation. As research continues to expand these capabilities, the promise of fully integrated, intelligent sensing, and communication systems in next-generation networks draws closer to realization.

## 3 Signal Design and Optimization

### 3.1 Signal Processing Techniques

Signal processing techniques play a vital role in the design of integrated systems that seamlessly handle both communication and sensing requirements. Within the context of Integrated Sensing and Communications (ISAC), the challenge lies in developing signal processing methods that can exploit the available resources to jointly optimize the system's dual functionalities. The continuous evolution of 5G and the anticipated transformative capabilities of 6G emphasize the need for innovative algorithmic approaches to meet these demands.

A critical aspect of signal design for ISAC systems involves the joint optimization of waveforms for simultaneous communication and sensing. Orthogonal Frequency Division Multiplexing (OFDM) is a popular scheme due to its robustness against multipath fading and spectrum efficiency. However, its application in simultaneous radar sensing and communication necessitates advanced waveform design techniques to mitigate interference and enhance performance [38]. To address the non-idealities intrinsic to multicarrier (MC) systems, researchers are exploring novel approaches such as Code-Division OFDM, which promises higher spectrum efficiency and robust interference management [39].

Time-frequency signal design significantly impacts ISAC system performance. Efficient utilization of the time-frequency plane is crucial for robust operations in dynamic environments. Optimized Time-Frequency Space (OTFS) modulation, an emerging technique, shows promise due to its ability to maintain robustness in highly mobile scenarios by providing delay-Doppler resilience. This technique's effectiveness in handling Doppler effects is pivotal for accurate sensing and stable communication links in fast-moving applications like vehicular networks and aerial platforms [40].

The role of Massive MIMO in integrated signal processing is another noteworthy development. By leveraging spatial degrees of freedom through spatial multiplexing and beamforming, Massive MIMO enhances both the resolution of sensing tasks and the capacity of communication channels [16; 3]. Its utility in improving spatial resolution is particularly advantageous in scenarios requiring detailed environmental mapping and precise target tracking. However, the high computational complexity and energy consumption associated with MIMO systems present ongoing challenges [16].

Compressed sensing (CS) techniques provide a strategic approach to managing sparse signal environments. CS enables signal acquisition below the Nyquist rate, making it an effective tool for efficiently reconstructing ISAC signals with reduced overhead [2]. These methods are being adapted for ISAC to enable efficient data collection and processing, notably in sensor networks and IoT applications where bandwidth and power are constrained.

Emerging trends in the field highlight the integration of artificial intelligence and machine learning (AI/ML) into signal processing workflows. AI/ML algorithms offer adaptive and predictive capabilities that can optimize resource allocation dynamically, enhance signal classification, and improve interference management [14; 22]. These approaches can significantly improve decision-making in environment-aware communications, translating into more meaningful and context-driven operations.

While significant progress has been made, the area of ISAC signal processing remains ripe with open challenges. Future explorations must address issues such as the standardization of signal formats, reducing computational complexity, and ensuring security in data exchange. A continued focus on the confluence of advanced signal processing, AI, and new computational paradigms will undoubtedly lead the charge in realizing the ambitious goals set out for 6G networks and beyond. This balanced amalgamation of technology and innovation is anticipated to redefine the landscape of wireless communications and sensing, promising unprecedented capabilities and enhanced user experiences.

### 3.2 Interference Management

In integrated sensing and communication (ISAC) systems, effective interference management is crucial for optimizing spectrum efficiency and ensuring stable operations across diverse applications. This subsection explores various methodologies for mitigating interference within these integrated systems, focusing on balancing the dual functionalities of communication and sensing.

Central to interference management is self-interference cancellation (SIC), a significant consideration in full-duplex communications where simultaneous transmission and reception occur on the same frequency channel. SIC employs advanced signal processing techniques to distinguish overlapping signals, effectively eliminating self-interference from transmitters and enhancing communication clarity [41]. Analog and digital cancellation play pivotal roles in this context; analog SIC involves pre-processing signals to cancel self-interference at the antenna level, while digital SIC applies advanced algorithms at the baseband.

Cross-link interference management is another key aspect, addressing interference between communication links or channels. Techniques like Non-Orthogonal Multiple Access (NOMA) enable multiple users to share the same frequency resource by distinguishing signals through power differences, thus optimizing spectrum use in densely deployed environments [42]. Cooperative interference strategies are also employed, where networks intelligently allocate resources to minimize inter-link interference, thereby enhancing overall network performance.

Dynamic spectrum management strategies are essential for adaptive interference mitigation, particularly in environments with fluctuating spectrum demands. These approaches use real-time data to dynamically allocate and reassign spectrum resources, effectively reducing conflicts between sensing and communication tasks. By leveraging reinforcement learning and other AI-driven techniques, networks can anticipate spectrum demands and adjust allocations preemptively, further enhancing spectrum efficiency [43].

Moreover, integrating machine learning (ML) and artificial intelligence (AI) into interference management is becoming increasingly common. Deep learning models can identify complex interference patterns and suggest optimal configurations for interference mitigation, thereby improving the robustness of integrated systems. As highlighted in [13], ML algorithms assist in extracting channel information to minimize interference impact.

Despite advancements in interference management, challenges remain, particularly in coordinating different layers of the network stack. Implementing cross-layer interference management solutions that integrate physical, MAC, and network layers holds promise for resolving these challenges but requires further investigation in real-world deployment scenarios.

As we move forward, research must continue to emphasize enhancing interference management techniques with adaptive capabilities to dynamically adjust to changing network conditions and demands. This includes advancing AI-driven strategies for real-time decision-making and optimizing resource allocation strategies for increasingly dense network deployments. The potential for innovations in interference management is substantial, offering vast improvements in the efficiency and reliability of integrated sensing and communication systems, and paving the way for more effective deployment in 5G-6G networks and beyond.

### 3.3 Energy Efficiency

The subsection on "Energy Efficiency" endeavors to explore the optimization strategies that enhance the power efficiency of integrated sensing and communication (ISAC) systems, crucial to balancing the dual demands on power consumption by both functionalities. In ISAC systems, reducing energy consumption while maintaining high performance standards remains a pivotal challenge, especially in the context of 5G and evolving 6G networks. The energy efficiency of ISAC systems is tightly coupled with their ability to effectively allocate power across communication and sensing tasks, thereby ensuring robust operational performance without excessive energy expenditure.

Power allocation optimization is a primary focus, as it involves strategic distribution of power resources among various functions of ISAC systems to achieve optimal system performance. Advanced algorithms are often employed to dynamically assign power based on instantaneous channel conditions, ensuring minimal energy waste. These include methods like stochastic optimization and convex relaxation, which allow for the fine-tuning of power distribution to meet sensing and communication requirements [10]. A critical aspect in this approach is ensuring a sustainable balance where the marginal utility of power in both communication and sensing is maximized.

Energy-aware network design, another crucial consideration, emphasizes architectural innovations that inherently reduce power usage. This includes implementing node on/off switching mechanisms, where network components are selectively powered down during low-demand periods, and employing virtualization technologies to minimize hardware dependence. These technologies enable ISAC systems to maintain operational effectiveness while significantly cutting energy consumption, illustrating a concerted move towards green networks [23]. The use of Massive MIMO technology is particularly beneficial in this context due to its ability to increase spectral efficiency, thereby allowing for more efficient power use [44].

Furthermore, the trade-offs between energy consumption and spectral efficiency are increasingly being scrutinized. There exists a delicate balance in optimizing these parameters, whereby improvements in spectral efficiency can often come at the cost of higher energy consumption as more sophisticated processing methods are employed. Addressing these trade-offs involves the deployment of intelligent algorithms that can predictively adjust system parameters to optimize both energy and spectrum usage dynamically [34]. The development of multi-objective optimization frameworks that simultaneously address multiple performance metrics such as reliability, throughput, and power efficiency is imperative for addressing this challenge.

Emerging trends point towards the integration of Machine Learning (ML) techniques that enhance predictive power management capabilities in ISAC systems. By leveraging ML, ISAC systems can anticipate network demands and adapt power usage in real-time to ensure that energy efficiency does not compromise system performance [34]. Such advancements stand to significantly bolster the energy efficiency of future ISAC deployments.

In summary, the pursuit of energy efficiency in ISAC systems is a multifaceted endeavor, involving strategic power allocation, energy-aware network designs, and intelligent trade-offs between spectral efficiency and power consumption. As the landscape of wireless technology continues to evolve, particularly towards 6G, emphasis on energy efficiency will remain a cornerstone, ensuring that ISAC systems meet future performance demands sustainably and efficiently.

### 3.4 Multi-Objective Optimization

In the rapidly evolving field of Integrated Sensing and Communication (ISAC) for 5G-6G networks, multi-objective optimization emerges as a pivotal element in balancing diverse system requirements such as data rate, reliability, latency, and energy efficiency. This subsection delves into the strategies employed to address these often conflicting objectives, enriched by both traditional and innovative approaches. 

The crux of multi-objective optimization in ISAC systems is effectively managing the trade-offs between various performance metrics. For example, maximizing data rates frequently clashes with demands for energy efficiency and low latency. A Pareto front methodology is commonly adopted to systematically explore these trade-offs, facilitating the creation of solutions that harmonize different objectives without excessively compromising any single one [45]. In this approach, a solution is deemed optimal when no improvement in one objective can be achieved without compromising another, allowing stakeholders to select configurations based on specific operational priorities.

A notable strategy in this context is cross-layer optimization, which accounts for interactions across different network layers. Rather than optimizing signal design at a single layer, this technique comprehensively aligns protocols and resources at the physical, MAC, and network layers. This integrative approach is indispensable in environments characterized by significant interference or diverse service demands [11]. Cross-layer optimization enables adaptive mechanisms to dynamically adjust to varying ISAC scenarios, thereby enhancing system robustness and performance.

The integration of machine learning and artificial intelligence into optimization strategies marks a transformative advancement, endowing systems with adaptive and predictive capabilities. AI methods, such as reinforcement learning, can dynamically navigate trade-offs in real-time, adjusting to shifts in network conditions and user needs [46]. This adaptability simplifies the management of multiple objectives and boosts system performance across an array of applications, from autonomous vehicles to smart city scenarios.

Nevertheless, several challenges remain. A critical issue is the computational complexity intrinsic to solving multi-objective problems, especially within dense network deployments featuring numerous parameters [47]. As networks scale, the solution space grows exponentially, demanding efficient algorithms and substantial computational resources to derive near-optimal solutions swiftly.

Emerging trends highlight the promise of hybrid optimization frameworks that amalgamate heuristic and analytical methods. These frameworks harness the expansive search capability of heuristics alongside the precision of analytical models, adeptly navigating large solution spaces [48]. Additionally, leveraging distributed optimization techniques, where various network segments compute localized solutions contributing to a global objective, can significantly diminish overhead and enhance scalability [32].

In conclusion, multi-objective optimization is a crucial research domain for ISAC systems, holding profound implications for future network design. As technologies advance, the necessity for sophisticated and adaptive optimization frameworks to reconcile varied conflicting objectives grows. Future endeavors should emphasize improving the scalability and integration of advanced AI-driven optimization methods to meet the increasing complexity of next-generation wireless networks [49].

### 3.5 Emerging Trends and Challenges

In the dynamic landscape of integrated sensing and communication (ISAC) systems, signal design and optimization are pivotal, driving technological advancements in 5G-6G networks. This subsection delves into the recent trends, challenges, and future directions in ISAC signal design, forming the nucleus of both communication and radar applications. Cutting-edge research asserts the integration of AI and machine learning as a transformative force, particularly in adaptive and predictive signal processing. AI-driven techniques are posited to significantly enhance spectrum efficiency and resource allocation, drawing from rich contextual datasets [34].

Emerging trends highlight a paradigm shift towards utilizing non-orthogonal multiple access (NOMA) and reconfigurable intelligent surfaces (RIS) to concurrently maximize sensing and communication performance. NOMA's potential to improve spectral efficiency and user access capabilities in ISAC systems is underscored by its advanced interference management techniques, which can accommodate a more extensive set of users simultaneously [26]. Simultaneously, RIS are instrumental in manipulating the wireless environment to augment signal propagation, enabling substantial gains in beamforming and minimizing interference [50].

Standardization efforts in ISAC signal design are gaining momentum, addressing the interoperability challenges posed by heterogeneous systems. Developing unified protocols is crucial to facilitate seamless integration across varied ISAC applications, ensuring compatibility and interoperability. However, the complexity of synchronizing diverse components and ensuring robust security frameworks remains a formidable challenge [51].

Practical implementation faces multi-faceted challenges, including hardware limitations and environmental interactions that can affect ISAC systems' performance. Innovations like carrier aggregation are suggested to enhance spectral utilization efficiency, enabling ISAC systems to dynamically adapt to spectrum fragmentation and optimize sensing performance across various bands [52]. This approach is complemented by strategies for energy-efficient designs, which are crucial given the rising demand for sustainable and eco-friendly communication technologies [37].

Concurrently, understanding the intricate trade-offs between communication and sensing is central to developing effective ISAC systems. Trade-offs often manifest in power allocation and spectrum partitioning, where objectives like maximizing data rates and minimizing interference converge [53]. Multi-objective optimization frameworks have emerged as valuable tools, enabling simultaneous achievement of diverse performance metrics such as enhanced reliability, reduced latency, and improved energy efficiency [54].

In conclusion, the evolving field of ISAC systems is driven by innovative signal design and optimization techniques, poised to redefine the next generation of wireless networks. Future research directions should focus on integrating advanced AI techniques for dynamic tuning of system parameters, further exploration of RIS-aided architectures, and resolving complex ISAC challenges related to hardware constraints and environmental adaptability. By overcoming these hurdles, ISAC systems can realize their full potential, playing a critical role in shaping the wireless communication landscape standards [6].

## 4 Applications and Use Cases

### 4.1 Autonomous and Connected Vehicles

The integration of sensing and communication systems in autonomous and connected vehicles represents a transformative shift in the automotive industry, promising enhancements in safety, navigation, and operational efficiencies. As vehicles transition from traditional designs to those embedded with advanced communication technologies, the potential for improved vehicle-to-everything (V2X) interactions is realized. This transition, underpinned by 5G and emerging 6G networks, provides a robust platform for the deployment of integrated sensing and communication (ISAC) systems, effectively addressing the need for reliable, real-time data exchange.

Central to the development of autonomous and connected vehicles is the capability to enhance safety features through collision avoidance and trajectory prediction. Integrated systems enable vehicles to continuously exchange data with surrounding infrastructure and other vehicles, providing a sophisticated level of situational awareness. For instance, the dual-functional framework designed for connected automated vehicles (CAVs) leverages 5G millimeter-wave technology to improve raw data sharing, illustrating the significant reduction in latency and increase in data rate that can be achieved [40]. Theoretically, this allows real-time adjustments to be made to vehicle trajectories to preclude accidents, demonstrating the actualization of collaborative sensing mechanisms.

The continuous advancement in ISAC technologies significantly contributes to optimizing navigation and path planning. This is achieved through precise location tracking and real-time environmental mapping, enabling vehicles to plan the most efficient routes with minimal travel time and energy consumption. The deployment of dense networks of sensors coupled with communication nodes empowers vehicles to dynamically adapt to changing road conditions and traffic patterns. By utilizing techniques such as multi-access channels and coordinated signal processing, vehicles harness comprehensive environmental data to drive decision-making processes [55].

In addition to safety and navigation, ISAC systems enhance communication efficiency by facilitating seamless V2X interactions. Achieving high communication throughput while maintaining low latency is crucial for effective vehicle coordination, especially in urban settings where traffic density is significant. Non-orthogonal multiple access (NOMA) techniques have been applied to ISAC frameworks to streamline communication processes, thereby enhancing the simultaneous handling of multiple signals and devices [10]. This innovation addresses the complexities inherent in signal interference management, achieving efficient spectrum utilization vital for supporting smart vehicular networks.

Comparing traditional and next-generation vehicle systems, the trade-offs in scalability, data fidelity, and operational complexity are evident. Enhanced sensing capabilities and improved communication fidelity must balance with the need for scalable solutions that can be integrated across diverse vehicular platforms. Emerging trends point to the adoption of AI-driven frameworks that facilitate semantic understanding of sensor data, thus aligning communication processes with actionable insights [11]. However, notable challenges remain in standardizing these systems, ensuring interoperability between different technologies and platforms, and addressing privacy concerns arising from increased data exchange.

Integrated sensing and communication systems are pivotal in driving the evolution of intelligent transportation systems. As the foundational underpinnings of autonomous vehicle networks are strengthened, the foreseeable future will witness further integration of cognitive machine intelligence and adaptive control systems, poised to revolutionize vehicular interaction paradigms. Future research will likely focus on refining these technologies, exploring optimal algorithms and frameworks to maximize efficiency while mitigating the risks associated with rapid technological advancements.

### 4.2 Smart City Infrastructure

Smart city infrastructure is experiencing a profound transformation driven by the advanced integration of sensing and communication technologies, enabled by the development and deployment of 5G and 6G networks. Within this context, Integrated Sensing and Communication (ISAC) systems emerge as pivotal components in fostering sustainable and efficient urban environments by enhancing decision-making capabilities and optimizing operational efficiencies.

A primary application of ISAC in smart cities is intelligent traffic management. By harnessing capabilities such as millimeter-wave (mmWave) communication, sensors are seamlessly integrated into traffic systems to provide real-time insights into vehicle and pedestrian flows [19]. This integration supports adaptive traffic signal controls that optimize flow, reduce congestion, and lower emissions. Furthermore, ISAC systems enable immediate incident detection and automated response coordination, significantly improving traffic safety and efficiency [56]. The complexity of these systems necessitates sophisticated signal processing techniques to handle substantial data throughput and processing demands, yet advancements in artificial intelligence help streamline data analysis and management tasks [35].

Environmental monitoring is another crucial aspect of smart city infrastructure enabled by ISAC. Through the expansive coverage and high-resolution capabilities of advanced communication systems, real-time air quality monitoring is achievable with high accuracy, facilitating proactive urban management strategies [17]. Sensor-equipped nodes distributed across urban areas can continuously relay data to centralized platforms where AI-driven analytics anticipate and mitigate ecological concerns [11]. The synergy between communication and sensing systems enables these monitoring devices to cover wider areas with less energy consumption by maximizing spectral efficiency and optimizing system design [57].

In terms of public safety and surveillance, ISAC fosters a more responsive and aware urban environment. Smart camera systems equipped with ISAC technologies can efficiently analyze foot traffic and detect anomalies through advanced image processing and radar analytics, aiding law enforcement and emergency response teams [58]. However, deploying such surveillance systems raises significant privacy concerns. Addressing these issues through enhanced encryption and secure data processing protocols is crucial to maintaining public trust and ensuring compliance with privacy regulations [59].

Despite the substantial benefits ISAC systems offer for smart city infrastructure, challenges remain in their widespread deployment. These challenges include high capital costs, the need for robust data security frameworks, and ensuring the interoperability of diverse systems and technologies [60]. Future research is expected to focus on overcoming these technical and operational barriers, developing smarter algorithms for data synthesis, and optimizing the trade-offs between sensing accuracy and communication efficiency [11]. Additionally, as we move toward 6G, the enhancement of ISAC functionalities through innovations like distributed and cell-free networks, holographic beamforming, and non-terrestrial deployments is anticipated. Such advancements will be critical in realizing the full potential of integrated technologies in building sustainable and resilient urban environments [61].

In conclusion, integrating sensing and communication within smart city infrastructure not only advances current technological capabilities but also offers novel solutions for sustainable urban management. Ongoing research and development will continue to push the boundaries of what smart cities can achieve, with ISAC technologies serving as a cornerstone of this transformation.

### 4.3 Healthcare and Remote Monitoring

The convergence of integrated sensing and communication (ISAC) technologies within the realm of healthcare signifies a paradigm shift in how medical services are delivered, particularly in telemedicine and remote monitoring. As 5G-6G networks mature, the potential for ISAC to transform healthcare systems is being increasingly realized, promising enhanced real-time data exchange and unprecedented patient care improvements.

ISAC technologies facilitate continuous monitoring of patients' vitals via connected devices, thereby enabling healthcare providers to conduct remote assessments and interventions with higher accuracy and reduced latency. The integration of massive MIMO and reconfigurable intelligent surfaces (RIS) plays a crucial role in these operations, enhancing signal processing capabilities to ensure reliable and efficient data transmission even in environments with high interference levels [44; 62]. These advanced antenna systems offer improved beamforming techniques that are pivotal in maintaining the robustness of telemedicine applications, where uninterrupted video and data exchange are critical.

In the context of remote patient monitoring, ISAC enables the deployment of Internet of Everything (IoE) environments that strategically integrate sensing devices to monitor health metrics, such as heart rate and glucose levels, in real-time. This functionality emphasizes the importance of intelligent reflection communication (IRC) technologies, inclusive of concepts like RIS and ambient backscatter communication, which collectively enhance data accuracy and security [30].

A notable advantage of ISAC-enabled healthcare solutions is their ability to balance spectrum and energy efficiencies while expanding coverage through strategic deployment of intelligent surfaces. Such advancements support the implementation of patient-centric models where dynamic environments adapt to individual health needs, optimizing therapeutic outcomes. The RIS can dynamically reconfigure communication pathways, ensuring that the network optimizes the signal-to-noise ratio for both communication and sensing tasks, thus enhancing the quality of telemedicine consultations [23; 62].

The integration of AI with ISAC systems further amplifies their efficacy in remote healthcare scenarios. By leveraging machine learning algorithms for signal processing, real-time data interpretation is enhanced, facilitating predictive analytics and early intervention strategies [22]. This synergy allows healthcare providers to anticipate potential adverse health events proactively, offering a significant advantage over traditional monitoring systems that predominantly rely on retrospective data analysis.

However, while the promise of ISAC in healthcare is immense, challenges persist, particularly in standardizing protocols to ensure interoperability across diverse medical devices and networks. As systems converge onto the ISAC framework, safeguarding patient data privacy becomes paramount, posing both technical and ethical challenges that necessitate comprehensive encryption and data protection measures [63].

Looking ahead, the deployment of ISAC systems in healthcare will likely expand to encompass more sophisticated applications, such as augmented reality (AR) in surgery and advanced robotic-assisted treatments, providing even greater precision and efficacy. Ensuring seamless integration and addressing the associated security and privacy challenges will be essential to fully realize the potential of ISAC in transforming healthcare delivery [35].

In summary, ISAC represents a transformative innovation in the healthcare landscape, offering robust solutions for telemedicine and remote monitoring. The synergistic interplay of advanced sensing, communication, and AI technologies is poised to redefine patient care, enabling proactive and personalized healthcare services. As the infrastructure supporting ISAC continues to evolve, its potential to enhance healthcare systems is boundless, subject to resolving key challenges in data handling and standardization.

### 4.4 Industrial and Manufacturing Applications

The advent of integrated sensing and communication (ISAC) technologies has ushered in a transformative era for industrial and manufacturing sectors, positioning them to significantly enhance process optimization through real-time data acquisition and smart decision-making. ISAC merges communication facilities with high-precision sensing, facilitating the seamless operation of industrial processes by improving diagnostics, predictability, and efficiency.

A key application of ISAC in industrial settings is predictive maintenance, which aims to prevent equipment failures before they occur, thereby minimizing downtime and maintenance costs. Continuous monitoring enabled by integrated systems allows critical asset parameters to be constantly tracked, identifying potential failures [7]. AI and machine learning algorithms play a crucial role in this process by pinpointing anomalies in data streams, predicting equipment malfunction with high accuracy, and optimizing repair schedules and resource allocation [46]. This not only boosts operational efficiency but also extends machinery life considerably, contributing to sustainable industrial practices.

Moreover, automation and robotics in manufacturing stand to benefit significantly from ISAC. The integration of sensing and communication capabilities allows for dynamic recalibration of robotic systems based on real-time environmental data, thereby enhancing precision and adaptability in manufacturing processes. Such systems are vital for tasks requiring high precision under varying operational conditions, promoting agile and flexible manufacturing environments. ISAC’s role is particularly pronounced in complex operations demanding intricate coordination between multiple robotic units, ensuring cohesive and efficient task execution [8].

Additionally, ISAC can greatly optimize supply chain and logistics operations by providing enhanced visibility and control over distributed processes. Real-time tracking, facilitated by sensing-communication integration, empowers industries to monitor inventory with high accuracy and coordinate deliveries efficiently [64]. These capabilities can reduce lead times, optimize warehouse operations, and ensure timely resource allocation throughout the supply chain. Furthermore, these advancements support the deployment of adaptive logistics networks that can quickly respond to unforeseen disruptions or changes in demand.

A comparative analysis of ISAC-enabled manufacturing versus traditional methods showcases notable efficiencies. Traditional systems often rely on segmented approaches for sensing and communication, which can lead to latency in information processing and a lack of process optimization. Conversely, ISAC fosters seamless integration that enhances operational coherence, reduces redundancies, and achieves precise coordination across the manufacturing scope [22]. However, implementing ISAC comes with challenges, including integration complexity, technical interoperability issues, and the need for robust cybersecurity measures to protect sensitive data flows [65].

The landscape of industrial ISAC applications is rapidly evolving, with emerging trends pointing towards more intelligent and autonomous systems. Future advancements are expected in multi-agent systems that utilize collaborative ISAC frameworks to harness distributed intelligence across industrial plants [66]. Moreover, the ongoing development of semantic communication systems tailored for industrial needs offers promising enhancements by ensuring goal-oriented, efficient data exchange and processing [67].

In conclusion, ISAC represents a transformative shift in industrial and manufacturing applications by enhancing efficiency, reducing operational costs, and catalyzing sustainable practices. Continued research and development efforts are vital in addressing existing challenges, refining current paradigms, and exploring novel opportunities in industrial ISAC applications. Thus, ISAC not only serves as an enabler of operational improvements but also stands as a cornerstone in the progression towards smart, sustainable industries.

### 4.5 Aerospace and Satellite Networks

The integration of sensing and communication within aerospace and satellite networks represents a transformative advancement in global connectivity and service delivery. As one of the frontiers of next-generation communication technologies, these integrated systems aim to enhance the operational efficiency of airborne platforms and extend connectivity to underserved regions. In the context of aerospace applications, the development of aeronautical communication systems focuses on improving data exchange between airborne platforms and ground control, thereby enhancing navigation and operational safety. Advanced signal processing techniques, such as those discussed in [68], are pivotal in enabling these high-frequency communications, which suffer from unique challenges related to propagation characteristics at higher altitudes and velocities.

Satellite networks play a critical role in global sensing and communication. The deployment of satellite-assisted sensing infrastructures facilitates comprehensive environmental monitoring, including weather prediction and natural disaster management. These satellite systems benefit from integrated approaches that utilize joint beamforming and power allocation strategies in non-terrestrial networks. The work on integrated systems helps satellite networks overcome inherent issues of signal interference and bandwidth constraints by employing coordinated beamforming techniques [69].

The integration of non-terrestrial networks extends the reach of communication services to remote and challenging environments, thus bridging the digital divide in underserved areas. Through the deployment of satellite and unmanned aerial vehicle (UAV) networks, integrated sensing and communication systems offer novel opportunities for connectivity and data acquisition in previously inaccessible zones. Such deployments make use of multi-objective optimization frameworks to manage the trade-offs between communication efficacy and energy utilization [54].

However, the deployment of these integrated systems in aerospace and satellite contexts involves significant trade-offs. On one hand, the sharing of bandwidth between radar sensing and communication signals necessitates the development of sophisticated interference management strategies, as highlighted in [70]. On the other hand, the energy constraints and size limitations intrinsic to airborne platforms demand the adoption of efficient resource allocation techniques to maximize operational longevity and system performance.

Emerging trends focus on the use of cognitive radio and machine learning to further optimize the performance of these integrative systems. Such technologies enable adaptive resource management based on environmental sensing data, allowing these systems to learn and predict optimal operational patterns, thus ensuring efficient spectrum use while maintaining high-quality data transmission [34].

A significant challenge lies in standardizing the protocols that govern sensing and communication integration. Harmonizing these elements across different geographies and technologies is essential for achieving seamless interoperability among aerospace and satellite networks. Efforts towards establishing unified standards for integrated sensing and communication are crucial in overcoming these challenges and facilitating the scalability of these technologies.

In conclusion, the amalgamation of sensing and communication capabilities within aerospace and satellite networks represents a paradigm shift in global connectivity. By resolving current challenges through innovative solutions and advanced technologies, such integrated systems hold the potential to revolutionize communication services and environmental monitoring on a global scale. The future direction will involve leveraging advancements in AI and machine learning to further enhance these systems' capabilities, thereby unlocking new opportunities for technological advancement and societal benefit.

## 5 Challenges and Solutions

### 5.1 Technical Barriers

The integration of sensing and communication functionalities in 5G-6G networks, while promising in terms of performance enhancements and adaptability, presents a series of technical challenges that need addressing. These challenges primarily involve hardware limitations, signal processing complexities, and environmental factors affecting system deployment and performance.

Hardware limitations in integrated systems often manifest in the form of size, weight, and power (SWaP) constraints, which pose significant challenges for deployment in devices such as unmanned aerial vehicles (UAVs) and base stations [64]. The demand for increased functionality without proportionate increments in these physical attributes pressures design paradigms and tests the limits of current manufacturing capabilities. The highly constrained SWaP envelope necessitates innovative approaches in hardware design, potentially leveraging advancements in nanotechnology and material sciences to develop compact and efficient components.

Signal processing complexities are another hurdle characterizing integrated sensing and communication systems. The demands of simultaneous data processing for both sensing and communication functions often result in high computational loads. This necessitates the development of advanced algorithms that can manage this dual functionality efficiently. The design of waveforms that optimally balance the needs of communication and sensing remains a non-trivial problem [5]. To address these issues, efforts have been made in employing compressed sensing techniques, which allow signal reconstruction at rates much below the conventional Nyquist rate, enhancing efficiency [2]. Furthermore, machine learning algorithms are being increasingly applied to optimize processing tasks, such as signal reconstruction and resource allocation, significantly enhancing system performance [14].

Environmental factors add an additional layer of complexity. Dense urban structures and rapidly changing environments cause signal blockages and interference that degrade both communication and sensing performance. In such challenging situations, techniques like reconfigurable intelligent surfaces (RISs) are employed to manipulate electromagnetic waves dynamically, enhancing signal quality [58]. Moreover, frequency and angle diversity, enabled by massive multiple-input multiple-output (MIMO) systems, can improve resilience against adverse environmental effects but introduce their challenges related to power consumption and signal processing complexity.

In addressing these barriers, researchers are exploring distributed architectures to effectively manage resource allocation, reduce processing bottlenecks, and enhance the adaptability of integrated systems. Distributed intelligent integrated sensing and communications (DISAC) approaches, for instance, propose a framework where sensing and communication operations are seamlessly managed across spatially distributed nodes, optimizing resource utilization and improving system agility [71]. The integration of artificial intelligence further bolsters these systems' decision-making capabilities, allowing dynamic adaptation to environmental conditions and operational demands [13].

As we advance towards the realization of 6G networks, ongoing research into overcoming these technical barriers is crucial. Emerging innovations in hardware design, advanced signal processing techniques, and AI-driven optimizations hold the potential to significantly enhance the feasibility and performance of integrated sensing and communication systems. These pedagogic strides are foundational to addressing real-world deployment constraints and ensuring that integrated systems achieve their full potential in modern digital ecosystems. However, continued interdisciplinary collaboration will be critical to surmount these complex technical hurdles and drive future advancements.

### 5.2 Privacy and Security

Integrated Sensing and Communication (ISAC) systems hold significant promise in meeting the bandwidth demands of next-generation networks, but they also introduce intricate privacy and security challenges due to the integration of sensing and communication tasks. This interconnectedness brings about new vectors for security breaches and privacy violations, necessitating comprehensive evaluation.

A prime concern is the potential for increased privacy issues, as ISAC systems depend on extensive data collection and sharing processes. The reliance on diverse data inputs from multiple sources elevates the risk of unauthorized access to sensitive information, such as location data and personal identifiers. Hence, the urgency to fortify physical layer security is paramount, ensuring robust data protection does not compromise system performance [59]. The sophistication of potential threats, including advanced eavesdropping techniques, calls for cutting-edge encryption and anonymization methods to safeguard personal data [59].

Additionally, ISAC systems are vulnerable to numerous security threats, such as network-based attacks like jamming and spoofing, which can disrupt both communication and sensing functions. The integration of these functionalities broadens the attack surface, thus necessitating resilient defense mechanisms to protect infrastructure integrity. In particular, attackers could exploit sensing data to misguide or deceive systems, leading to vulnerabilities in critical applications like autonomous driving or industrial automation [15; 58].

To counter these threats, ISAC systems need to continually refine their security paradigms with advanced encryption techniques and secure key management protocols. Implementing quantum-safe communications shows promise, as these techniques can significantly diminish the feasibility of decryption by brute force, particularly crucial in light of advancing computational power [59]. Moreover, integrating Artificial Intelligence (AI) and machine learning in security strategies could greatly enhance ISAC networks by predicting and identifying potential threats, thereby enabling dynamic adaptation and preemptive correction measures [65].

A comparative analysis of potential security mechanisms reveals trade-offs between latency, system complexity, and robustness. While conventional encryption methods may offer a high level of security, they often come at the cost of increased processing times and latency, potentially hindering the real-time functionality of ISAC systems. Conversely, lightweight encryption protocols provide expediency but may offer reduced protection, highlighting the need for a balanced approach [5].

Future focus should be on developing robust, adaptable security frameworks that can evolve alongside technological advancements in ISAC systems. This approach emphasizes the importance of collaborative efforts across disciplines to establish standardized protocols that ensure the protection of user data and system operations without compromising performance. Overcoming these challenges requires concerted initiatives from academia, industry, and regulatory bodies, fostering innovation while prioritizing privacy and security [11]. As ISAC technology continues to advance, these security measures must scale and adapt to effectively safeguard the burgeoning network ecosystem.

### 5.3 Standardization and Interoperability

Integrated Sensing and Communication (ISAC) systems necessitate a rigorous standardization framework to ensure seamless interoperability across diverse network infrastructures and achieve optimal performance. This subsection elucidates the critical role of establishing standardized protocols in facilitating the integration of sensing and communication capabilities within 5G-6G networks.

The landscape of wireless communication is rapidly evolving, with ISAC systems emerging as a pivotal innovation for next-generation networks. As these systems aim to amalgamate sensing capabilities, such as radar and localization, with communication functionalities, the demand for unified standards is more pronounced than ever. The complexity inherent in heterogeneous networks, compounded by diverse frequency bands and equipment, underscores the need for a collaborative framework to ensure effective cross-system operability [7].

One of the primary approaches to achieving standardization in ISAC systems is the development of unified communication standards that seamlessly integrate sensing functionalities into existing network architectures. This involves defining protocols that cater to the dual nature of ISAC systems and accommodate the technical specificities of both communication and sensing tasks. For instance, standards must address how bandwidth is allocated dynamically between sensing and communication tasks to maximize resource utilization without compromising on performance [22]. This calls for innovative spectrum sharing techniques such as those explored in NOMA-based systems [10] where the superimposed signals serve dual purposes.

Interoperability challenges are further amplified by the diversity of technologies employed by different network operators. Achieving seamless operation demands a cohesive effort to harmonize these differences through standardized interfaces and protocols. An example can be seen in deploying Reconfigurable Intelligent Surfaces (RISs), which require specific standardizations to ensure consistent and harmonious operation across devices and networks [72]. Moreover, the integration of emerging innovations such as the massive MIMO and RISs presents additional interoperability hurdles due to the necessity of synchronizing array processing and phase shifts across platforms [62; 44].

In addressing these challenges, industry-wide collaboration becomes indispensable. Efforts such as those undertaken in the RISE-6G project exemplify how international cooperation can lead to the establishment of standardized practices and innovation that enable dynamic and adaptable ISAC systems [24]. Collectively, such efforts can pave the way for universally accepted standards that foster interoperability while facilitating the advancement of future networks.

Looking forward, continued research and dialogue in standardization are essential. Emerging trends like the inclusion of AI-driven signal processing and edge computing in ISAC systems necessitate a reevaluation of current standards to accommodate these cutting-edge technologies [31]. Furthermore, ensuring robust privacy and security measures within ISAC standards will be crucial to securing sensitive data exchanged across these integrated networks, as highlighted by various security-focused efforts [63].

In synthesis, achieving standardization and interoperability in ISAC systems is pivotal for the effective operation of next-generation networks. It requires a coordinated approach that integrates diverse technological advances into a cohesive standard framework. As the field continues to evolve, a concerted effort towards harmonization will be essential in realising the full potential of ISAC systems in 5G-6G networks, ultimately enhancing network capabilities and service delivery.

## 6 Evaluation and Future Directions

### 6.1 Evaluation Metrics for ISAC Systems

In advancing Integrated Sensing and Communication (ISAC) systems in next-generation networks, evaluating their effectiveness and efficiency requires robust and comprehensive metrics. This paper synthesizes key performance indicators that drive this assessment, focusing on communication efficiency, sensing capabilities, and integration efficacy.

To initiate, communication performance metrics such as latency, data rate, and capacity are paramount for gauging the efficiency of ISAC systems. Latency, in particular, is critical because it influences real-time processing capabilities, which is vital for applications such as autonomous vehicles and industrial automation [40]. High data rates facilitate the seamless exchange of large data volumes necessary for integrated systems; in these high-capacity environments, such throughput ensures the simultaneous delivery of communication and sensing functionalities [40].

Sensing performance metrics are equally crucial, with sensing accuracy and range resolution serving as primary indicators. Sensing accuracy determines the precision of target detection, which translates directly into system effectiveness. Range resolution measures an ISAC system's ability to distinguish between multiple targets within close proximity, a feature essential for applications like unmanned aerial vehicle (UAV) navigation and high-resolution mapping [3]. Sensing mutual information, which quantifies the shared information between sensing sequences, further elucidates the interaction quality between sensing and communication [73].

Further extending the evaluation metrics beyond individual functions, interoperability and integration indicators such as signal-to-interference-plus-noise ratio (SINR) and coverage probability are vital. These metrics assess the seamless operation between sensing and communication, essential for maintaining high system reliability amid complex environments [32]. SINR serves as a critical factor in determining the quality of received signals, impacting the clarity and fidelity of the data exchanged between sensors and communications frameworks. Coverage probability indicates the spatial extent to which ISAC systems can effectively maintain these integrated services, reflecting their operational robustness under diverse scenarios [64].

Comparative analysis of these metrics reveals trade-offs that illustrate the strengths and limitations of different ISAC approaches. A primary challenge lies in balancing high sensitivity in sensing against the necessity for rapid, high-throughput communication, especially in congestion-prone environments [71]. Emerging trends in machine learning and artificial intelligence (AI) propose enhancements to these metrics by optimizing resource allocation dynamically and predicting network conditions with higher accuracy [14]. AI-driven techniques promise to foresee demand and adjust resources accordingly, improving not only individual metric outcomes but also their synergistic operation.

Future research directions indicate the importance of further refining these metrics to adapt to evolving technological paradigms. This includes investigating multi-objective optimization for simultaneous enhancement across communication and sensing dimensions and integrating new performance indicators such as those related to energy efficiency and ecological impact [22]. Developing a comprehensive evaluation framework that synthesizes these multi-faceted metrics will be indispensable for realizing the full potential of ISAC systems, ensuring their robustness, scalability, and efficacy in addressing the diverse demands of modern and future network applications. As the boundary between communication and sensing continues to blur, it is pivotal to align metrics with the complex interplay of these functionalities, paving the way for profound advancements in ISAC system design and implementation.

### 6.2 Optimization Techniques in ISAC

Optimization techniques in Integrated Sensing and Communication (ISAC) systems are pivotal for enhancing both communication throughput and sensing accuracy, ensuring these systems can successfully leverage shared spectral resources while addressing challenges such as interference management, energy efficiency, and real-time adaptability. In this subsection, we will explore various optimization strategies employed in ISAC, delineating their strengths, limitations, and potential future challenges, and how they interlink with broader trends discussed earlier.

Signal design and waveform optimization are core aspects of ISAC, focusing on maximizing the utility of limited spectral bandwidth for both sensing and communication tasks. Waveform design strategies aim for joint optimization to simultaneously meet the demands of high-resolution sensing and robust communication. Advanced designs, such as mutual information-maximizing waveforms, play a crucial role by fine-tuning waveform characteristics to align with specific environment and application needs [5]. This not only enhances spectral efficiency but also aids in effective interference management, a common challenge in shared spectrum environments [74].

Resource allocation strategies form another critical component of ISAC optimization. Given the fluctuating demand across sensing and communication functionalities, dynamic and adaptive allocation of resources like power, bandwidth, and time slots becomes essential. Techniques such as stochastic geometry and game theory offer sophisticated frameworks for modeling and solving these allocation problems. Game-theoretic approaches, in particular, enable strategic decision-making under competitive conditions, maximizing overall system utility while ensuring fairness among users [65]. However, real-time application complexity can impose computational overheads, thus necessitating lightweight algorithms capable of timely execution [75].

Effectively managing interference remains a core challenge in integrating sensing and communication activities. Advanced beamforming and power control, leveraging massive multiple-input multiple-output (MIMO) systems, are critical strategies for tackling this issue [16]. These involve optimizing beamforming vectors and adjusting power levels to target energy transmissions towards desired goals while mitigating cross-interference, all the while maintaining service quality. This adaptability is often enhanced through machine learning algorithms, which predict interference patterns and adjust system parameters [13].

Emerging trends suggest that integrating artificial intelligence can further refine the optimization processes within ISAC systems. AI-driven techniques allow for adaptive optimization by incorporating predictive analytics and real-time decision-making processes. Machine learning models, for instance, can improve signal processing and resource management by learning from historical data to predict future system states [22]. However, this integration also presents challenges concerning computational costs, model training time, and generalization capabilities [76].

In conclusion, the evolving optimization techniques for ISAC systems underscore a need for multidisciplinary approaches that combine advanced mathematical modeling, AI, and cutting-edge hardware capabilities. Looking ahead, hybrid strategies integrating classical optimization with AI should be explored, focusing on reducing computational loads and enhancing real-time adaptability. Addressing these challenges is crucial for the feasible deployment and performance optimization of ISAC in next-generation networks, paving the way for seamless, efficient, and intelligent communication and sensing systems. This naturally bridges the discussion to the subsequent integration of AI into ISAC systems, as detailed in the following section, where AI's role in advancing performance and adaptability in future 6G infrastructure is further explored.

### 6.3 Integration of Artificial Intelligence in ISAC

The integration of artificial intelligence (AI) in Integrated Sensing and Communication (ISAC) systems enhances their performance and adaptability, addressing the complex demands of future 6G networks. AI technologies, such as machine learning and deep learning, bring profound advancements in optimizing resource allocation, waveform design, and interference management, critical to the development of robust ISAC systems.

AI-driven signal processing in ISAC offers significant improvements in performance optimization. Machine learning algorithms facilitate adaptive signal design, improving the integration of communication and sensing functionalities by effectively predicting and adjusting to dynamic environmental conditions. For instance, deep learning models are increasingly employed to process and interpret complex signal patterns, enhancing the accuracy and efficiency of sensing activities [77]. These technologies capitalize on data-driven approaches to refine system parameters continuously, allowing ISAC systems to autonomously learn and adapt over time.

Intelligent resource management through AI is pivotal in negotiating the trade-offs between communication and sensing tasks. AI algorithms, capable of real-time decision-making, forecast demand fluctuations and allocate resources dynamically, optimizing both energy consumption and spectral efficiency. Techniques such as reinforcement learning are instrumental in this context, providing the ability to predict resource necessities and adjust allocations promptly, thus maintaining a harmonious balance between the distinct requirements of ISAC components [34].

AI also facilitates the development of semantic and goal-oriented frameworks in ISAC, transforming conventional data processing methodologies. By utilizing semantic-informed processing, AI enhances the extraction of meaningful insights from raw data, aligning the operational goals of ISAC systems with user-centric application requirements. This shift towards goal-oriented communications and sensing not only boosts system efficacy but also significantly reduces data processing overheads, thereby improving overall network throughput [78].

Despite these advancements, the integration of AI in ISAC is not without challenges. The need for large datasets to train AI models poses significant barriers, particularly in scenarios where data scarcity or proprietary constraints exist. Furthermore, the computational complexity associated with AI model training necessitates formidable processing power, often placing additional demands on existing infrastructure [78].

Emerging trends suggest an increasing focus on developing lightweight AI models capable of functioning under stringent resource constraints. Techniques such as transfer learning and online learning offer potential solutions by leveraging previously acquired knowledge to enhance model efficiency and reduce training requirements. Moreover, the integration of AI within decentralized ISAC architectures promises to unlock further potential in distributed intelligence, enhancing network scalability and robustness [24].

Looking ahead, the role of AI in ISAC will continue to expand, driven by the growing needs of next-generation networks and the push towards more intelligent, adaptive systems. Future research is expected to delve into the development of more sophisticated AI frameworks that can seamlessly integrate with evolving ISAC technologies, driving efficiency and innovation in this transformative domain [22]. By advancing the synergy between AI and ISAC, future networks can achieve unparalleled levels of performance and adaptability, paving the way for new horizons in wireless communication systems.

### 6.4 Future Research Directions and Technological Innovations

The ongoing evolution of Integrated Sensing and Communication (ISAC) within 5G and 6G networks heralds numerous promising research directions and technological innovations poised to enhance system performance and broaden application possibilities. This subsection highlights these advancements, focusing on technological innovations, theoretical challenges, and practical implications, thus complementing the previously discussed AI integration and preceding the exploration of technical and regulatory hurdles.

A pivotal research thrust is the creation of energy-efficient ISAC systems, aiming to optimize power use while sustaining high communication and sensing capabilities. In light of the escalating energy demands of dense networks and the drive for sustainable progress, innovations such as intelligent reflecting surfaces (IRS) and non-orthogonal multiple access (NOMA) become crucial. IRS can dynamically manipulate the wireless environment to improve signal quality and spectrum efficiency, thereby conserving energy [79; 30; 31]. Meanwhile, NOMA provides a framework for managing simultaneous communication and sensing tasks, thereby optimizing spectral usage and reducing power requirements [28; 80].

Advanced ISAC architectural designs are another critical area of focus. Novel architectures like distributed and hybrid systems present opportunities for developing scalable and efficient network solutions. Distributed architectures utilize connected elements to coordinate sensing and communication functions more effectively, enhancing coverage and reducing latency [81; 49]. Hybrid approaches leverage edge and cloud computing to dynamically manage data processing, facilitating real-time responses and minimizing backhaul demands [82; 48].

AI integration into ISAC systems further advances their capabilities by addressing dynamic spectrum management, optimizing resource allocation in real-time, and boosting semantic communications through learning mechanisms [46; 83]. Machine learning facilitates transition to semantic- and goal-oriented frameworks that emphasize data relevance over volume [67; 47; 22].

An innovative research direction is embedding ISAC systems with Digital Twin (DT) technology, creating precise virtual models of physical systems for real-time performance optimization. Digital Twins enhance ISAC by coordinating sensing, computation, and communication, thus anticipating network demands and adjusting operations accordingly [84; 85]. This aligns with the larger vision of a comprehensive and intelligent Internet of Everything (IoE) ecosystem.

However, challenges such as standardization, privacy, and security persist. Standardization efforts must design protocols ensuring interoperability across various technologies and applications [6; 86]. Protecting data privacy and ensuring secure communications become crucial, especially with AI and cloud computing deployment, where user data is vulnerable [87; 88]. Techniques like secure multi-party computation and homomorphic encryption are vital in addressing these concerns [58].

In summary, while ISAC systems for 5G and 6G networks face unique challenges, the technological innovations considered here hold the promise to revolutionize connectivity. By pursuing research on energy efficiency, architectural design, AI integration, and Digital Twins, ISAC can transition into an era of sustainable, smart, and adaptive networks. As the field advances, collaboration between industry and academia is essential to transition these innovations from theory to practice, ultimately transforming network interactions with both digital and physical realms.

### 6.5 Challenges and Open Issues in ISAC

Integrated Sensing and Communication (ISAC) systems are promising for efficiently utilizing spectrum resources and achieving seamless integration of communication with sensing capabilities. However, several technical and regulatory challenges must be addressed before ISAC can be widely implemented. This subsection discusses these challenges, presenting an academic analysis of the existing literature while identifying crucial open issues that future research must tackle.

From a technical perspective, one of the most significant challenges is the proper allocation of spectrum resources. ISAC systems must balance the needs of communication and sensing without causing detrimental interference. The integration of sensing and communication has to overcome traditional methods of spectrum division and instead adopt techniques such as cognitive radio and machine learning for dynamic spectrum allocation [34]. However, designing systems that can efficiently manage spectrum without compromising on performance remains a central challenge.

Hardware limitations present another substantial technical barrier. There is a need for specialized hardware that can handle the dual demands of sensing and communication simultaneously. Current devices often face constraints in processing power and efficiency, limiting their capacity for widespread ISAC deployment. Developments in hardware such as full-duplex transceivers are being explored to address these issues, allowing simultaneous transmit and receive operations [89].

The integration of reconfigurable intelligent surfaces (RIS) poses both an opportunity and a challenge. While RIS can enhance communication and sensing by dynamically altering the propagation environment [50], the control and optimization of RIS in real-time for integrated operations require further investigation. Optimizing the placements and configurations of these surfaces in dynamic environments is necessary to fully harness their potential [79].

Additionally, ISAC systems must navigate the privacy and security issues inherent in their architecture. The shared nature of ISAC, while beneficial for resource efficiency, also presents increased vulnerability to security threats such as unauthorized interception of sensitive data [51]. Encryption techniques and secure protocols must be advanced to mitigate these risks effectively.

Regulatory challenges also impede the standardization of ISAC systems. The development of universal protocols that facilitate interoperability across diverse networks and devices is critical. The variability in international regulations, compounded by the fast-paced evolution of technology, means there is a pressing need for global standards that can accommodate various ISAC implementations [90]. Collaborative international efforts and industry consensus on standardization protocols are essential steps forward.

In conclusion, tackling these challenges requires a multidisciplinary approach that encompasses advances in signal processing, machine learning, and hardware design. By addressing these issues, ISAC systems can reach their full potential, offering enhanced service capabilities for future wireless networks. Future research must focus on optimizing spectrum-sharing mechanisms, developing robust security protocols, and achieving regulatory harmony to ensure the successful adoption of ISAC technologies. These efforts will significantly contribute to realizing intelligent and efficient next-generation communication systems.

## 7 Conclusion

In synthesizing the comprehensive landscape of Integrated Sensing and Communication (ISAC) for 5G and 6G networks, this survey reveals a field characterized by rapid technological advancements and multifaceted applications. The seamless integration of sensing with communication harbors the potential to fundamentally transform network architectures and service offerings. In striving for this transformative impact, a myriad of technologies and conceptual frameworks have emerged, each affording unique strengths and posing distinct challenges. These encompass innovative signal processing techniques [2], multi-objective optimizations [1], and the utilization of non-orthogonal multiple access (NOMA) strategies [10].

The academic exploration identifies the promising confluence of high-frequency communications such as millimeter-wave and terahertz bands that facilitate the vast data requirements of ISAC systems [61; 3]. In lockstep, enabling technologies like Massive MIMO, reconfigurable intelligent surfaces, and AI-driven advances are pivotal in augmenting both sensing fidelity and communication efficacy [91; 13].

Through this broad survey, pivotal strengths such as efficient spectrum utilization and system resource sharing are highlighted as significant merits of ISAC [5]. However, these advancements are tempered by technical hurdles, including hardware constraints, signal processing complexities, and the burgeoning need for standardized protocols to ensure interoperability [15].

The comparative analysis underlines definitive trade-offs in maximizing dual function effectiveness, such as the balance between sensing accuracy and communication throughput, a challenging equilibrium requiring innovative design solutions [92]. Additionally, the integration of artificial intelligence stands as a promising avenue for advancing ISAC systems by enabling adaptive and predictive management of network resources [11].

Emerging trends point toward the expansive role of semantic and goal-oriented communication frameworks that promise to revolutionize data processing and system operations by focusing on the meaning and intent of transmitted data—a marked departure from traditional data-centric paradigms [67]. This transition aligns with the broader trajectory of 6G objectives, which envision networks as enablers of not merely communication but comprehensive data sharing ecosystems capable of high-precision localization and sensing tasks [17].

In reflecting on these technological trajectories and theoretical underpinnings, it becomes evident that sustained research efforts are paramount. Addressing existing roadblocks hinges on interdisciplinary collaborations poised to leverage emerging opportunities in high-frequency spectrums and complex system integrations. The future of ISAC lies in its potential to redefine how networks perceive, interact with, and adapt to their environments—a pursuit that demands ongoing innovation and cross-sector synergy [35; 71]. As we progress toward realizing the full potential of ISAC in 6G networks, the alignment of academic, industrial, and regulatory initiatives will be crucial in harnessing the complete spectrum of opportunities this field has to offer.

## References

[1] Integrated Sensing, Computation, and Communication  System Framework and  Performance Optimization

[2] Applications of Compressed Sensing in Communications Networks

[3] A Vision of 6G Wireless Systems  Applications, Trends, Technologies, and  Open Research Problems

[4] Framework for an Innovative Perceptive Mobile Network Using Joint  Communication and Sensing

[5] Integrated Sensing and Communication Signals Toward 5G-A and 6G  A  Survey

[6] Towards Integrated Sensing and Communications for 6G

[7] Integrated Sensing and Communications  Recent Advances and Ten Open  Challenges

[8] Joint Communication, Sensing and Computation enabled 6G Intelligent  Machine System

[9] NOMA Inspired Interference Cancellation for Integrated Sensing and  Communication

[10] NOMA Empowered Integrated Sensing and Communication

[11] Integration of Communication and Sensing in 6G  a Joint Industrial and  Academic Perspective

[12] Toward Immersive Communications in 6G

[13] AI Empowered Channel Semantic Acquisition for 6G Integrated Sensing and  Communication Networks

[14] The Convergence of Machine Learning and Communications

[15] Practical Issues and Challenges in CSI-based Integrated Sensing and  Communication

[16] Joint Communication and Sensing  Models and Potential of Using MIMO

[17] 6G White Paper on Localization and Sensing

[18] Millimeter Wave Communications for Future Mobile Networks

[19] A Survey of Millimeter Wave (mmWave) Communications for 5G   Opportunities and Challenges

[20] Beyond 5G RIS mmWave Systems  Where Communication and Localization Meet

[21] Unlocking Potentials of Near-Field Propagation: ELAA-Empowered Integrated Sensing and Communication

[22] Integrated Sensing and Communication for 6G  Ten Key Machine Learning  Roles

[23] Reconfigurable Intelligent Surfaces  Potentials, Applications, and  Challenges for 6G Wireless Networks

[24] Wireless Environment as a Service Enabled by Reconfigurable Intelligent  Surfaces  The RISE-6G Perspective

[25] A Survey on Integrated Sensing and Communication with Intelligent  Metasurfaces  Trends, Challenges, and Opportunities

[26] Exploiting NOMA and RIS in Integrated Sensing and Communication

[27] Orthogonal Time Frequency Space Modulation -- Part III  ISAC and  Potential Applications

[28] NOMA for Integrating Sensing and Communications towards 6G  A Multiple  Access Perspective

[29] Deep Learning-based Design of Uplink Integrated Sensing and  Communication

[30] Intelligent Reflection Enabling Technologies for Integrated and Green  Internet-of-Everything Beyond 5G  Communication, Sensing, and Security

[31] Deep-Learning Channel Estimation for IRS-Assisted Integrated Sensing and  Communication System

[32] Cooperative Sensing and Communication for ISAC Networks  Performance  Analysis and Optimization

[33] 6G Networks  Beyond Shannon Towards Semantic and Goal-Oriented  Communications

[34] Intelligent Wireless Communications Enabled by Cognitive Radio and  Machine Learning

[35] Integrating Sensing, Computing, and Communication in 6G Wireless  Networks  Design and Optimization

[36] Joint Design of Overlaid Communication Systems and Pulsed Radars

[37] Energy Efficient Beamforming Optimization for Integrated Sensing and  Communication

[38] An Experimental Proof of Concept for Integrated Sensing and  Communications Waveform Design

[39] Code-Division OFDM Joint Communication and Sensing System for 6G  Machine-type Communication

[40] Design and Performance Evaluation of Joint Sensing and Communication  Integrated System for 5G MmWave Enabled CAVs

[41] Millimeter Wave communication with out-of-band information

[42] Cooperation in 5G HetNets  Advanced Spectrum Access and D2D Assisted  Communications

[43] Towards 6G Evolution  Three Enhancements, Three Innovations, and Three  Major Challenges

[44] Using Massive MIMO Arrays for Joint Communication and Sensing

[45] Five Facets of 6G  Research Challenges and Opportunities

[46] Artificial Intelligence-Enabled Intelligent 6G Networks

[47] Twelve Scientific Challenges for 6G  Rethinking the Foundations of  Communications Theory

[48] Joint Beamforming and Offloading Design for Integrated Sensing,  Communication and Computation System

[49] Joint Communication and Sensing for 6G -- A Cross-Layer Perspective

[50] RIS-Assisted Communication Radar Coexistence  Joint Beamforming Design  and Analysis

[51] Robust and Secure Resource Allocation for ISAC Systems  A Novel  Optimization Framework for Variable-Length Snapshots

[52] Carrier Aggregation Enabled Integrated Sensing and Communication Signal  Design and Processing

[53] Joint Spectrum Partitioning and Power Allocation for Energy Efficient Semi-Integrated Sensing and Communications

[54] Multi-Objective Signal Processing Optimization  The Way to Balance  Conflicting Metrics in 5G Systems

[55] Integrated Sensing and Communication in Coordinated Cellular Networks

[56] Millimeter Wave Vehicular Communication to Support Massive Automotive  Sensing

[57] Potential Key Technologies for 6G Mobile Communications

[58] Role of Sensing and Computer Vision in 6G Wireless Communications

[59] Security and privacy for 6G  A survey on prospective technologies and  challenges

[60] Challenges & Solutions for above 6 GHz Radio Access Network Integration  for Future Mobile Communication Systems

[61] 6G Wireless Communication Systems  Applications, Requirements,  Technologies, Challenges, and Research Directions

[62] Joint Communication and Radar Sensing with Reconfigurable Intelligent  Surfaces

[63] Secure Intelligent Reflecting Surface Aided Integrated Sensing and  Communication

[64] UAV-Enabled Integrated Sensing and Communication  Opportunities and  Challenges

[65] Integrated Sensing, Computation and Communication in B5G Cellular  Internet of Things

[66] Cooperative ISAC Networks  Performance Analysis, Scaling Laws and  Optimization

[67] Semantic Communications for Future Internet  Fundamentals, Applications,  and Challenges

[68] An Overview of Signal Processing Techniques for Millimeter Wave MIMO  Systems

[69] Joint Active and Passive Beamforming Design for Reconfigurable  Intelligent Surface Enabled Integrated Sensing and Communication

[70] Radar and Communication Co-existence  an Overview

[71] Distributed Intelligent Integrated Sensing and Communications  The  6G-DISAC Approach

[72] Reconfigurable Intelligent Surface for Physical Layer Security in  6G-IoT  Designs, Issues, and Advances

[73] Integrated Sensing and Communications  A Mutual Information-Based  Framework

[74] Multicarrier ISAC: Advances in Waveform Design, Signal Processing and Learning under Non-Idealities

[75] 6G Enabled Advanced Transportation Systems

[76] Environment Semantic Communication  Enabling Distributed Sensing Aided  Networks

[77] Deep-Learning-Based Channel Estimation for IRS-Assisted ISAC System

[78] Integrated Sensing, Communication, and Computation Over-the-Air  MIMO  Beamforming Design

[79] The Rise of Intelligent Reflecting Surfaces in Integrated Sensing and  Communications Paradigms

[80] NOMA-aided Joint Communication, Sensing, and Multi-tier Computing  Systems

[81] Towards Distributed and Intelligent Integrated Sensing and  Communications for 6G Networks

[82] Digital-Twin-Enabled 6G  Vision, Architectural Trends, and Future  Directions

[83] Toward Ambient Intelligence  Federated Edge Learning with Task-Oriented  Sensing, Computation, and Communication Integration

[84] Integrated Communication, Localization, and Sensing in 6G D-MIMO  Networks

[85] Integrated Sensing and Communication Driven Digital Twin for Intelligent  Machine Network

[86] Interference Management for Integrated Sensing and Communication  Systems  A Survey

[87] Semantic Revolution from Communications to Orchestration for 6G: Challenges, Enablers, and Research Directions

[88] Understand-Before-Talk (UBT)  A Semantic Communication Approach to 6G  Networks

[89] Exploiting Self-Interference Suppression for Improved Spectrum  Awareness Efficiency in Cognitive Radio Systems

[90] Radio Resource Management in Joint Radar and Communication  A  Comprehensive Survey

[91] Enabling Joint Communication and Radar Sensing in Mobile Networks -- A  Survey

[92] Fundamental Detection Probability vs. Achievable Rate Tradeoff in  Integrated Sensing and Communication Systems

