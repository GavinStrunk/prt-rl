"""
DAgger: Dataset Aggregation from Demonstrations
===============================================

DAgger is an imitation learning method that leverages an expert to provide the true label for the best action to take in a given state. The action labeling procedure can be done using a human expert or another expert algorithm. Using the expert, DAgger proceeds to learn in a supervised fashion.

.. pcode::
   :linenos:

    % This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
    \begin{algorithm}
    \caption{Quicksort}
    \begin{algorithmic}
    \PROCEDURE{Quicksort}{$A, p, r$}
        \IF{$p < r$}
            \STATE $q = $ \CALL{Partition}{$A, p, r$}
            \STATE \CALL{Quicksort}{$A, p, q - 1$}
            \STATE \CALL{Quicksort}{$A, q + 1, r$}
        \ENDIF
    \ENDPROCEDURE
    \PROCEDURE{Partition}{$A, p, r$}
        \STATE $x = A[r]$
        \STATE $i = p - 1$
        \FOR{$j = p$ \TO $r - 1$}
            \IF{$A[j] < x$}
                \STATE $i = i + 1$
                \STATE exchange
                $A[i]$ with     $A[j]$
            \ENDIF
            \STATE exchange $A[i]$ with $A[r]$
        \ENDFOR
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}

"""
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.collectors import SequentialCollector
from prt_rl.common.policies import DistributionPolicy
from prt_rl.common.distributions import Categorical, Normal


@dataclass
class DAggerConfig:
    """
    Hyperparameter Configuration for the DAgger agent.

    Tuning tips & tricks are found in the description of each parameter:

    Args:
        buffer_size (int): Size of the aggregated dataset (expert-labeled) kept for training.
            • What it does: Caps how much (s,a, expert-a*) history you retain across DAgger
              iterations. Too small ⇒ you forget early states; too large ⇒ slower epochs and
              potential drift if early data is low quality.
            • Start here:
                - Low-dim control (CartPole, Pendulum, locomotion state vectors): 50k–200k.
                - Image observations (Atari, camera inputs): 100k–1M (images are redundant; you can
                  downsample frames or use every k-th frame to achieve 200k–500k unique states).
            • Increase if: you see covariate-shift oscillations (policy fixes one failure mode,
              regresses on old ones), or your validation loss on early data rises over time.
            • Decrease if: each training epoch is very slow and your learning plateaus; prune old
              data using a FIFO buffer, class-balanced reservoir, or prioritize recent failure cases.

        batch_size (int): Number of samples drawn from the buffer per optimizer "outer" update.
            • What it does: Controls gradient noise; bigger batch ⇒ smoother but costlier step.
            • Start here:
                - Low-dim: 1k–4k.
                - Images: 4k–16k effective samples per outer update (often achieved via multiple mini-batches).
            • Increase if: loss is very noisy across updates or policy is unstable between DAgger
              iterations.
            • Decrease if: GPU RAM is tight or you want faster, more frequent updates; compensate
              with a slightly lower learning rate if you reduce batch size a lot.
            • Tip: If you use gradient accumulation, effective_batch = mini_batch_size * accum_steps.

        learning_rate (float): Optimizer step size (typically Adam/AdamW).
            • Start here:
                - MLP on low-dim: 1e-3 (Adam).
                - CNN/ResNet encoder frozen: 1e-3 for head; if fine-tuning encoder, 1e-4 (head 10× higher).
                - ViT/CLIP fine-tuning: 1e-5–3e-5 (head 10× higher).
            • Increase if: loss decreases too slowly and you see no overfitting signs after several epochs.
            • Decrease if: training loss diverges, spikes after adding new DAgger data, or eval accuracy
              on expert labels worsens while train loss falls (classic too-big LR).
            • Scaling: roughly linear with batch size (halve LR if you halve effective batch).

        optim_steps (int): Number of optimizer steps (outer updates) per DAgger iteration.
            • What it does: How much you train on the freshly aggregated data before the next
              on-policy rollout + querying the expert again.
            • Start here:
                - Small buffers (<100k): 1–5; Medium (100k–500k): 5–20; Large (≥500k): 20–100.
            • Increase if: the policy doesn’t incorporate new corrections before the next rollout
              (you keep making the same mistakes between iterations).
            • Decrease if: overfitting to the latest batch (performance on a held-out slice of older
              data drops), or wall-clock becomes dominated by training instead of data collection.
            • Tip: Early-stop within each iteration using a small validation split from the buffer.

        mini_batch_size (int): Per-step sub-batch for the optimizer (used inside each outer update).
            • What it does: Controls memory footprint and gradient noise inside one outer update.
            • Start here: 32–256 for low-dim; 64–512 for images (depending on GPU RAM).
            • Increase if: gradients are very noisy and you can afford more memory.
            • Decrease if: you hit OOM or want higher update frequency; adjust LR if you change this a lot.

        max_grad_norm (float): Gradient clipping threshold (global norm).
            • What it does: Prevents rare large updates when the aggregated dataset shifts (a common
              DAgger event).
            • Start here: 1.0–5.0 for vision models; 5.0–10.0 for small MLPs.
            • Decrease if: you observe occasional loss explosions after appending new expert data.
            • Increase (or disable) if: training is very slow and gradients are consistently tiny
              (check gradient norms first; don’t increase by default).
            • Tip: Log gradient norms; clipping should rarely activate. If it triggers often, revisit LR.

    Additional DAgger-specific guidance (not separate args here but crucial):
        • Expert query budget / β-schedule:
            - Start with high expert intervention (β≈1: mostly expert actions) and anneal to 0
              over 5–20 iterations. Faster anneal on simple, deterministic tasks; slower on
              stochastic, high-dim ones.
            - If your learned policy destabilizes when β falls, slow the anneal and/or increase
              optim_steps for a couple iterations.

        • Data selection:
            - Prefer querying the expert on states visited by the current policy (on-policy),
              optionally biased toward high-uncertainty or high-loss states.
            - Balance classes/actions if your domain is imbalanced; otherwise the buffer can
              overrepresent easy states.

        • Regularization:
            - Weight decay (AdamW): 1e-4 for MLPs; 0–5e-5 for fine-tuning pretrained vision backbones.
            - Label smoothing (0.0–0.05) can help with noisy experts.
            - Data augmentation (RandAug/ColorJitter) helps vision, but keep it mild to avoid
              drifting from the expert’s visual domain.

        • Monitoring:
            - Track supervised loss on (i) new DAgger slice, (ii) a held-out historical slice.
              If (i) ↓ while (ii) ↑, you’re overfitting to the new slice → reduce optim_steps or
              add replay mixing that oversamples older hard examples.
            - Periodically roll out with β=0 to estimate true closed-loop performance.

    Reasonable presets:
        • Low-dim control (MLP):
            buffer_size=100_000, batch_size=2_000, mini_batch_size=128,
            learning_rate=1e-3, optim_steps=5, max_grad_norm=5.0

        • Vision with frozen encoder (train head only):
            buffer_size=300_000, batch_size=8_000, mini_batch_size=256,
            learning_rate=1e-3 (head), optim_steps=20, max_grad_norm=1.0

        • Vision with fine-tuning:
            buffer_size=500_000, batch_size=8_000–16_000, mini_batch_size=256–512,
            learning_rate=1e-4 (backbone), 1e-3 (head), optim_steps=20–50, max_grad_norm=1.0
    """
    buffer_size: int = 10000
    batch_size: int = 1000
    learning_rate: float = 1e-3
    optim_steps: int = 1
    mini_batch_size: int = 32
    max_grad_norm: float = 10.0

class DAgger(BaseAgent):
    r"""
    Dataset Aggregation from Demonstrations (DAgger) agent.

    Examples:
        .. code-block:: python

            from prt_rl import DAgger, DAggerConfig
            from prt_rl.common.policies import DistributionPolicy

            # Setup Environment
            # env = ...

            # Load the expert policy and experience buffer
            expert_policy = SB3Agent(model_dir=str(expert_dir), model_type='ppo', env_name=env_name, device=device)
            experience_buffer = ReplayBuffer.load(str(expert_dir / "ppo_expert_experience.pkl"), device=device)

            # Configure hyperparameters
            config = DAggerConfig(buffer_size=5000, batch_size=500, learning_rate=1e-3)

            # Create DAgger Policy
            policy = DistributionPolicy(env_params, distribution=Categorical)

            # Create DAgger Agent
            agent = DAgger(env_params=env.get_parameters(), expert_policy=expert_policy, experience_buffer=experience_buffer, config=config)

            # Train
            agent.train(env=env, total_steps=10000)

    Args:
        policy (DistributionPolicy | None): The policy to be used by the agent. If None, a default policy will be created based on the environment parameters.
        expert_policy (BaseAgent): The expert agent to provide actions for the states.
        experience_buffer (ReplayBuffer): The replay buffer to store experiences.        
        device (str): Device to run the agent on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self,
                 expert_policy: BaseAgent,
                 experience_buffer: ReplayBuffer,                 
                 policy: DistributionPolicy,
                 config: DAggerConfig = DAggerConfig(),
                 device: str = 'cpu',
                 ) -> None:
        super(DAgger, self).__init__()
        self.expert_policy = expert_policy
        self.experience_buffer = experience_buffer
        self.config = config
        self.device = torch.device(device)

        self.policy = policy
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.loss_function = self._get_loss_function(self.policy)

    @staticmethod
    def _get_loss_function(policy: DistributionPolicy) -> torch.nn.Module:
        """
        Returns the loss function used for training the policy based on the type of distribution.

        Args:
            policy (DistributionPolicy): The policy for which to get the loss function.
        Returns:
            torch.nn.Module: The loss function to be used for training.
        Raises:
            ValueError: If the distribution type is not supported.
        """
        if issubclass(policy.distribution, Categorical):
            # For categorical distributions, use CrossEntropyLoss
            return torch.nn.CrossEntropyLoss()
        elif issubclass(policy.distribution, Normal):
            # For continuous distributions, use MSELoss
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported distribution type {policy.distribution.__class__} loss function.")
    
    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Perform an action based on the current state using the policy.
        
        Args:
            state: The current state of the environment.
        
        Returns:
            The action to be taken by the policy.
        """
        with torch.no_grad():
            return self.policy(state, deterministic=deterministic)
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],              
              logger: Optional[Logger] = None,
              evaluator: Evaluator = Evaluator(),
              show_progress: bool = True
              ) -> None:
        """
        Train the DAgger agent using the provided environment and expert policy.

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        # Resize the replay buffer with size: initial experience + total_steps
        self.experience_buffer.resize(new_capacity=self.experience_buffer.size + self.config.buffer_size)

        # Add initial experience to the replay buffer
        collector = SequentialCollector(env=env, logger=logger)

        num_steps = 0

        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect experience using the current policy
            policy_experience = collector.collect_experience(policy=self.policy, num_steps=self.config.batch_size)
            num_steps += policy_experience['state'].shape[0]

            # Get expert action for each state in the collected experience
            expert_actions = self.expert_policy(policy_experience['state'])

            # Update the policy experience with expert actions
            policy_experience['action'] = expert_actions

            # Add the policy experience to the replay buffer
            self.experience_buffer.add(policy_experience)

            # Optimize the policy
            losses = []
            for _ in range(self.config.optim_steps):
                for batch in self.experience_buffer.get_batches(batch_size=self.config.mini_batch_size):
                    # Compute the loss between the policy's actions and the expert's actions
                    if issubclass(self.policy.distribution, Categorical):
                        policy_logits = self.policy.get_logits(batch['state'])
                        loss = self.loss_function(policy_logits, batch['action'].squeeze(1))
                    else:
                        policy_actions = self.policy(batch['state'])
                        loss = self.loss_function(policy_actions, batch['action'])

                    losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.max_grad_norm)
                    self.optimizer.step()
            
            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, "
                                                                   f"Loss: {np.mean(losses):.4f},")

            # Log the training progress
            if logger.should_log(num_steps):
                for scheduler in schedulers:
                    logger.log_scalar(name=scheduler.parameter_name, value=getattr(scheduler.obj, scheduler.parameter_name), iteration=num_steps)
                logger.log_scalar(name='loss', value=np.mean(losses), iteration=num_steps)

            # Evaluate the agent periodically
            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()
