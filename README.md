# Group_Presentation
This repo is used to do the presentation some paper shared in group meeting

## 2023-10-09
Paper: Autonomous UAV Navigation: A DDPG-based Deep Reinforcement Learning Approach

[Online Resources](https://ieeexplore.ieee.org/abstract/document/9181245)


### Background & Purpose
**Background**:
1. UAV plays more and more important role in Smart City 
   1. Cargo delievery
   2. Traffic monitoring
   3. ![Figure_1]()
2. Challenges:
   1. Path finding/planning
   2. Most of the existing solutions are based on computationally complex mixed-integer linear programming or evolutionary algorithms that do not always achieve near-optimal solutions. 
      1. In addition, most of the existing approaches are centralized, which limits the ability of the system to handle real-time problems and increases the communication overhead between the central node and the flight units.

**Purpose**:
1.  This paper propose a framework:
    1.  Unlike existing RL-based solutions that typically operate in discretized environments, this framework aims to provide autonomous navigation for UAVs in continuous action space to reach fixed or moving targets dispersed in a 3D spatial region, taking into account the safety of the UAV.
    2.  The approach uses a Deep Deterministic Gradient Descent (DDPG) based methodology with the aim of allowing the UAV to determine the optimal route to accomplish its mission safely, i.e. avoiding obstacles.
    3.  In the training phase, a transfer learning approach is used to first train the UAV on how to reach its destination in an obstacle-free environment, and then the learned model is used in other environments with specific obstacle locations so that the UAV learns how to navigate to its destination avoiding obstacles.



### DDPG

### Simulation results

### Inspiration & Reflection