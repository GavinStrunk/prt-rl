import torch
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvironmentInterface 
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.collectors import SequentialCollector


class DAgger(BaseAgent):
    def __init__(self,
                 policy: torch.nn.Module,
                 buffer_size: int = 10000,
                 mini_batch_size: int = 32,
                 ) -> None:
        self.policy = policy
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform an action based on the current state using the policy.
        
        Args:
            state: The current state of the environment.
        
        Returns:
            The action to be taken.
        """
        with torch.no_grad():
            return self.policy(state)
    
    def train(self,
              env: EnvironmentInterface,
              expert_policy: torch.nn.Module,
              experience_buffer: ReplayBuffer,
              total_steps: int,
              ) -> None:
        # Resize the replay buffer with size: initial experience + total_steps
        experience_buffer.resize(new_capacity=experience_buffer.size + self.buffer_size)

        # Add initial experience to the replay buffer
        collector = SequentialCollector(env=env)

        for i in range(total_steps):
            # Collect experience using the current policy
            policy_experience = collector.collect_experience(policy=self.policy, num_steps=1)

            # Get expert action for each state in the collected experience
            expert_actions = expert_policy(policy_experience['state'])

            # Update the policy experience with expert actions
            policy_experience['action'] = expert_actions

            # Add the policy experience to the replay buffer
            experience_buffer.add(policy_experience)

            # Optimize the policy
            for _ in range(self.optim_steps):
                for batch in experience_buffer.get_batches(batch_size=self.mini_batch_size):
                    policy_actions = self.policy(batch['state'])

                    # Compute the loss between the policy's actions and the expert's actions
                    loss = torch.nn.functional.mse_loss(policy_actions, batch['action'])

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                    self.optimizer.step()
