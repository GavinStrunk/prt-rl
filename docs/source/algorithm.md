# Implementing a New Agent

## Step 1: Define Config Dataclass

```python
@dataclass
class AlgorithmConfig:
    params: int

```

## Step 2: Define PolicySpec Dataclass

```python
@dataclass
class PPOPolicySpec:
    backbone: BackboneSpec = MLPBackboneSpec()
```

## Step 3: Define the Policy class
The PolicyModule interface defines the minimum API that is required by the collectors to gather samples from the environment using the policy. The algorithm specific policy is intended to extend this base API to add methods required for the agent algorithm. For example, a PPO agent requires an evaluation_action method that returns the log probability and entropy given a state and action. This class is a contract between external policies and what is required by the agent for training.

```python
class PPOPolicy(PolicyModule):

```

## Step 4: Create PolicyFactory
The policy factory builds an algorithm specific policy given a policy spec and a policy class. The policy factory ensures a valid policy is constructed, so the honerous is not on the user to handle details like ensuring the activation is compatible with an actor distribution. 

Therefore, the purpose of the factory is to safely build policies so the user is not meant to build policies directly when using the algorithm interface.

## Step 5: Implement the Algorithm Agent
Time for the meat and potatos. 