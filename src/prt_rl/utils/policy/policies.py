import importlib
from abc import ABC, abstractmethod
from typing import Any
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams

def load_from_mlflow(
        tracking_uri: str,
        model_name: str,
        model_version: str,
) -> 'Policy':
    """
    Loads a model that is either registered in mlflow or associated with a run id.

    Args:
        tracking_uri (str): mlflow tracking uri
        model_name (str): name of the model in the registry
        model_version (str): string version of the model

    Returns:
        Policy: policy object
    """
    try:
        import mlflow
    except ModuleNotFoundError:
        raise ModuleNotFoundError("mlflow is required to be installed load a policy from mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_registered_models()
    for model in registered_models:
        print(f"Model Name: {model.name}")

    model_str = f"models:/{model_name}/{model_version}"
    policy = mlflow.pyfunc.load_model(model_uri=model_str)

    # Extract the metadata
    metadata = policy.metadata.metadata

    # Policy factory
    module_name = f"prt_rl.utils.policy"
    try:
        module = importlib.import_module(module_name)
        policy_class = getattr(module, metadata['type'])
        policy = policy_class.load_from_dict(metadata['policy'])
    except ModuleNotFoundError:
        raise ValueError(f"Class {metadata['type']} not found")

    # if metadata['type'] == 'QTablePolicy':
    #     return QTablePolicy.load_from_dict(metadata['policy'])
    return policy

class Policy(ABC):
    """
    Base class for implementing policies.

    Args:
        env_params (EnvParams): Environment parameters.
        device (str): The device to use.
    """
    def __init__(self,
                 env_params: EnvParams,
                 device: str = 'cpu',
                 ) -> None:
        self.env_params = env_params
        self.device = device

    @abstractmethod
    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        """
        Chooses an action based on the current state. Expects the key "observation" in the state tensordict

        Args:
            state (TensorDict): current state tensordict

        Returns:
            TensorDict: tensordict with the "action" key added
        """
        raise NotImplementedError

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a key value parameter

        Args:
            name (str): name of the parameter
            value (Any): value to set
        """
        pass


    def save_to_dict(self) -> dict:
        """
        Serializes the policy into a dictionary.
        Child classes should override this method if they are capable of being saved.
        """
        return {}











