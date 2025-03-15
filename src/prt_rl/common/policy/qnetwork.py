from dataclasses import asdict
from typing import Any, Optional
import torch
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy.policies import Policy
import prt_rl.common.networks as qnets
import prt_rl.common.decision_functions as dfuncs

class QNetworkPolicy(Policy):
    """
    QNetwork policy is an ANN based q value function approximation.

    Args:
        env_params (EnvParams): environment parameters
        num_envs (int): number of environments
        decision_function (DecisionFunction): decision function. If None (default), EpsilonGreedy is used with an epsilon of 0.1.
        qnetwork (torch.nn.Sequential, optional): QNetwork. If None (default), an MLP QNetwork will be created.
        device (str): String device name. Default is 'cpu'.
    """
    def __init__(self,
                 env_params: EnvParams,
                 num_envs: int = 1,
                 decision_function: Optional[dfuncs.DecisionFunction] = None,
                 qnetwork: Optional[torch.nn.Sequential] = None,
                 device: str = 'cpu'
                 ) -> None:
        super(QNetworkPolicy, self).__init__(env_params=env_params, device=device)
        self.num_envs = num_envs

        if qnetwork is None:
            self.q_network = qnets.MLP(
                state_dim=self.env_params.observation_max+1,
                action_dim=self.env_params.action_max+1,
            )
        else:
            self.q_network = qnetwork

        if decision_function is None:
            self.decision_function = dfuncs.EpsilonGreedy(epsilon=0.1)
        else:
            self.decision_function = decision_function

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        state = state['observation']
        q_values = self.q_network.forward(state)
        action = self.decision_function.select_action(q_values)
        state['action'] = action
        return state

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in QNetworkPolicy.")

    @classmethod
    def load_from_dict(cls, data: dict) -> 'QNetworkPolicy':
        """
        Constructs a QNetworkPolicy from a dictionary. It is assumed the data dictionary was saved with the save_to_dict method.

        Args:
            data (dict): Dictionary of QNetworkPolicy parameters

        Returns:
            QNetworkPolicy: QNetwork policy object
        """
        env_params = EnvParams(**data['env_params'])

        # Dynamically load the decision function
        decision_function_class = getattr(dfuncs, data['decision_function']['type'])
        decision_function = decision_function_class.from_dict(data['decision_function'])

        # Load the QNetwork dynamically
        q_network_class = getattr(qnets, data['q_network_class'])
        q_network = q_network_class(**data['q_network_init_args'])
        q_network.load_state_dict(data['q_network'])

        policy = cls(
            env_params=env_params,
            num_envs=data['num_envs'],
            decision_function=decision_function,
            qnetwork=q_network,
            device=data['device']
        )
        return policy

    def save_to_dict(self) -> dict:
        """
        Serializes the QNetworkPolicy into a dictionary to it can be saved.

        Returns:
            dict: Dictionary of QNetworkPolicy parameters and values needed to load it.
        """
        return {
            'env_params': asdict(self.env_params),
            'num_envs': self.num_envs,
            'decision_function': self.decision_function.to_dict(),
            'q_network_class': self.q_network.__class__.__name__,
            'q_network_init_args': self.q_network.init_args(),
            'q_network': self.q_network.state_dict(),
            'device': self.device,
        }