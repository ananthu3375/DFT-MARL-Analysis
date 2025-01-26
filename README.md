# **Analysis of Adversarial Multi-Agent Game Outcomes of Dynamic Fault Tree Games** #

The purpose of this research project is to analyze the adversarial multi-agent game outcomes of Dynamic Fault Tree (DFT) games to identify and prioritize the vulnerabilities in a system, that can impact the system fault tolerance. By doing this, the aim is to create a model that can enhance the system reliability and safety. Modern engineering systems demand complexity, and while dealing with complexity, ensuring system reliability and safety is a critical goal in Model-Based Systems Engineering, particularly in safety-critical environments. This study explores the use of Adversarial Multi-Agent Reinforcement Learning to analyze and improve system reliability with a prototype system modeled using a Dynamic Fault Tree (DFT). A zero-sum game is implemented with two adversarial agents, the fault-injecting Red Agent and the fault-repairing Blue Agent, trained using Double Deep Q-Networks within a Multi-Agent Reinforcement Learning environment. The Red Agent strategically injects faults into system components located at the base level known as Basic Events, while the Blue Agent aims to repair failures and maintain system operability.

The adversarial agents interact in a turn-based sequence, with rewards structured to incentivize system failure for the Red Agent or prolonged operation for the Blue Agent. The overall experiment is conducted in the PettingZoo multi-agent library. The trained models are put to the test by playing several sets of games, where the outcomes of these games are analyzed, resulting in a basic decision to improve system reliability. Event Usage, Improvement Potential, and Reliability analysis are conducted using the Red Agent’s policy to identify the critical events. The analysis aims to highlight frequently targeted events and event sequences, including combinations of Basic Events, which significantly influence fault propagation and Top Event failure.

The study further evaluates the critical events and introduces modifications to the initial DFT model structure to see how system performance can be enhanced through targeted interventions. The agents are retrained over the new DFT, and the results are analyzed to understand the importance of improvements in system fault tolerance. Redundancy and parameter adjustments have a significant impact on the fault tolerance of a system. The results confirm the proposed method significantly improves system reliability by making it more tolerant to adversarial faults. Moreover, the game win percentage shows improvement over the modified DFT, which implies that the Adversarial Multi-Agent Reinforcement Learning framework can be implemented to identify critical components in a system. Improving the identified critical events enhances system reliability. Ultimately, this work underscores the potential of combining reinforcement learning with DFT modeling for more effective system risk management and provides a foundation for future developments in system reliability engineering.

## **Lower Dimensional DFT** 
![LD_DFT](https://github.com/user-attachments/assets/7dcbfe7a-ebaa-4175-a9d6-c7387aaff304)

## **Higher Dimensional DFT**
![HD_DFT](https://github.com/user-attachments/assets/979e9f3e-eba2-43bd-846c-bac5251c1ca6)

## **Simulated Fault Propagation in the Lower Dimensional DFT**
![ld_fault_propagation_animation](https://github.com/user-attachments/assets/e9621695-e441-47c5-8334-77bb45439849)

## **Simulated Fault Propagation in the Higher Dimensional (Improved) DFT**
![hd_fault_propagation_animation](https://github.com/user-attachments/assets/cec4dbb5-ceea-4394-af62-a7c31229d7a5)
