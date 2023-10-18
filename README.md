# Group_Presentation
This repo is used to do the presentation some paper shared in group meeting

## 2023-10-09
Paper: Autonomous UAV Navigation: A DDPG-based Deep Reinforcement Learning Approach

### Background & Purpose
**Background**:
1. UAV plays more and more important role in Smart City 
   1. Cargo delievery
   2. Traffic monitoring
   3. ![Figure_1]()
2. Challenges:
   1. Path finding/planning
   2. Most of the existing solutions are based on computationally **complex mixed-integer linear programming or evolutionary algorithms** that do not always achieve near-optimal solutions. 
      1. In addition, most of the existing approaches are **centralized**, which **limits the ability** of the system to **handle real-time problems** and **increases the communication overhead** between the central node and the flight units.

**Purpose**:
1.  This paper propose a framework:
    1.  Unlike existing RL-based solutions that typically operate in discretized environments, this framework aims to provide autonomous navigation for UAVs in continuous action space to reach fixed or moving targets dispersed in a 3D spatial region, taking into account the safety of the UAV.
    2.  The approach uses a **Deep Deterministic Gradient Descent (DDPG) based methodology** with the aim of allowing the UAV to determine the optimal route to accomplish its mission safely, i.e. avoiding obstacles.
    3.  In the training phase, a transfer learning approach is used to first train the UAV on how to reach its destination in an obstacle-free environment, and then the learned model is used in other environments with specific obstacle locations so that the UAV learns how to navigate to its destination avoiding obstacles.



### DDPG

Deep Deterministic Policy Gradient (DDPG), an off-policy Reinforcement Learning algorithm.

Before DDPG, we need to know some other things:
1. SPG
2. DPG
3. DQ
4. DDPG

### SPG - Stochastic Policy Gradient

#### Notations: 
- $\pi_\theta(s, a)$: This represents the policy, which gives the probability of taking action \( a \) given state \( s \).
- $Q^w(s, a)$: This is the action-value function, representing the expected return for taking action \( a \) in state \( s \) and then following policy \($\pi$\).
- $\nabla_\theta$: This denotes the gradient with respect to \( $\theta$ \).

#### Formula Explanation:
- $\nabla_\theta J(\theta)$: This is the gradient of the policy's objective function \( J \) with respect to its parameters \( \theta \). The goal is to optimize this objective function using gradient ascent.
- $\mathbb{E}$: This is the expectation, averaging over the contributions of all possible state-action pairs.
- $\nabla_\theta \log \pi_\theta(s, a)$: This is the gradient of the logarithm of the policy with respect to \( $\theta$ \). This term provides insights into how to tweak the parameter \( $\theta$ \) to adjust the probability of taking action \( a \) in state \( s \).
- $Q^w(s, a)$: The action-value function provides guidance to the policy, indicating which actions are preferable in a given state.

The essence of the formula is that we aim to increase the probability of actions that yield high returns and decrease the probability of actions that lead to low returns. Therefore, we use the action-value function \( $Q^w(s, a)$ \) to weight the policy gradient to adjust the policy.

This gradient-based approach offers an effective mechanism to explore the optimal policy space, iteratively refining the policy parameters \( $\theta$ \) to hone in on the optimal policy.


#### DPG
The "grad objective function" of an algorithm is:
- the quantity that describes how the value of the objective function varies with the model parameters
- and we can update and optimize the model parameters based on this information.

### Simulation results

### Inspiration & Reflection

### Links
Resources & References:
1. [Paper: Autonomous UAV Navigation: A DDPG-based Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/9181245)
2. [Deep Deterministic Policy Gradient(DDPG) — an off-policy Reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/deep-deterministic-policy-gradient-ddpg-an-off-policy-reinforcement-learning-algorithm-38ca8698131b)
3. 
