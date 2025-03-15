from dataclasses import asdict
import io
from tensordict.tensordict import TensorDict
import torch
from typing import Any, Optional
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy.policies import Policy
import prt_rl.common.qtable as qtabs
import prt_rl.common.decision_functions as dfuncs

class QTablePolicy(Policy):
    """
    A Q-Table policy combines a q-table action value function with a decision function.

    Args:
        env_params (EnvParams): environment parameters
        num_envs (int): number of environments
        decision_function (DecisionFunction): decision function. If None (default), EpsilonGreedy is used with an epsilon of 0.1.
        qtable (QTable, optional): Q-Table. If None (default), Q-Table will be created with initial values of 0 and no visit tracking.
        device (str): String device name. Default is 'cpu'.

    """
    def __init__(self,
                 env_params: EnvParams,
                 num_envs: int = 1,
                 decision_function: Optional[dfuncs.DecisionFunction] = None,
                 qtable: Optional[qtabs.QTable] = None,
                 device: str = 'cpu'
                 ):
        super(QTablePolicy, self).__init__(env_params=env_params, device=device)
        assert env_params.action_continuous == False, "QTablePolicy only supports discrete action spaces."
        assert env_params.observation_continuous == False, "QTablePolicy only supports discrete observation spaces."

        self.num_envs = num_envs

        if qtable is None:
            self.q_table = qtabs.QTable(
                state_dim=self.env_params.observation_max +1,
                action_dim=self.env_params.action_max +1,
                batch_size=num_envs,
                initial_value=0.0,
                track_visits=False,
                device=device,
            )
        else:
            self.q_table = qtable

        if decision_function is None:
            self.decision_function = dfuncs.EpsilonGreedy(epsilon=0.1)
        else:
            self.decision_function = decision_function

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        obs_val = state['observation']
        q_values = self.q_table.get_action_values(obs_val)

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
            raise ValueError(f"Parameter '{name}' not found in QTablePolicy.")

    def get_qtable(self) -> qtabs.QTable:
        """
        Returns the Q-Table used in the policy.

        Returns:
            QTable: Q-Table
        """
        return self.q_table

    @classmethod
    def load_from_dict(cls, data: dict) -> 'QTablePolicy':
        """
        Constructs a QTablePolicy from a dictionary. It is assumed the data dictionary was saved with the save_to_dict method.

        Args:
            data (dict): Dictionary of QTablePolicy parameters

        Returns:
            QTablePolicy: Q-Table policy object
        """
        env_params = EnvParams(**data['env_params'])

        # Dynamically load the decision function
        decision_function_class = getattr(dfuncs, data['decision_function']['type'])
        decision_function = decision_function_class.from_dict(data['decision_function'])

        # Deserialize q_table from binary data
        q_table_buffer = io.BytesIO(data["q_table"])
        q_table_data = torch.load(q_table_buffer)

        # Deserialize visit table from binary data if it exists
        if data['visit_table'] is not None:
            visit_table_buffer = io.BytesIO(data['visit_table'])
            visit_table_data = torch.load(visit_table_buffer)
        else:
            visit_table_data = None

        # Dynamically create QTable class and load qtable and visit table
        q_table_class = getattr(qtabs, data['q_table_class'])
        q_table = q_table_class(**data['q_table_init_args'])
        q_table.q_table = q_table_data
        q_table.visit_table = visit_table_data

        # Construct the QTablePolicy
        policy = cls(
            env_params=env_params,
            num_envs=data['num_envs'],
            decision_function=decision_function,
            qtable=q_table,
            device=data['device']
        )
        return policy

    def save_to_dict(self) -> dict:
        """
        Serializes the QTablePolicy into a dictionary to it can be saved.

        Returns:
            dict: Dictionary of QTablePolicy parameters and values needed to load it.
        """
        # Serialize the q_table tensor to binary data
        q_table_buffer = io.BytesIO()
        torch.save(self.q_table.q_table, q_table_buffer)
        q_table_buffer.seek(0)

        # Serialize the visit table tensor to binary data if there is one
        visit_table = None
        if self.q_table.track_visits:
            visit_table_buffer = io.BytesIO()
            torch.save(self.q_table.visit_table, visit_table_buffer)
            visit_table_buffer.seek(0)
            visit_table = visit_table_buffer.getvalue()

        return {
            'env_params': asdict(self.env_params),
            'num_envs': self.num_envs,
            'decision_function': self.decision_function.to_dict(),
            'q_table_class': self.q_table.__class__.__name__,
            'q_table_init_args': self.q_table.init_args(),
            'q_table': q_table_buffer.getvalue(),
            'visit_table': visit_table,
            'device': self.device,
        }