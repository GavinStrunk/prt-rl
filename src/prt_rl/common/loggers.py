import json
import mlflow
import mlflow.pyfunc
import os
import shutil
import tempfile
import torch
from typing import Optional
from prt_rl.common.policies import BasePolicy


class Logger:
    """
    Based class for implementing loggers for RL algorithms.

    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, type_: str, **kwargs):
        if type_ not in cls._registry:
            raise ValueError(f"Unknown media reader type: {type_}")
        return cls._registry[type_](**kwargs)
    
    def close(self):
        """
        Performs any necessary logger cleanup.
        """
        pass

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        """
        Logs a dictionary of parameters. Parameters are values used to initialize but do not change throughout training.

        Args:
            params (dict): Dictionary of parameters.
        """
        raise NotImplementedError("log_parameters must be implemented in subclasses.")
    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None,
                   ) -> None:
        """
        Logs a scalar value. Scalar values are any metric or value that changes throughout training.

        Args:
            name (str): Name of the scalar value.
            value (float): Value of the scalar value.
            iteration (int, optional): Iteration number.
        """
        raise NotImplementedError("log_scalar must be implemented in subclasses.")
    def save_policy(self,
                    policy: BasePolicy,
                    ) -> None:
        """
        Saves the policy to the logger.

        Args:
            policy (BasePolicy): Policy to save.
        """
        raise NotImplementedError("save_policy must be implemented in subclasses.")
    def save_agent(self,
                    agent: object,
                    ) -> None:
        """
        Saves the agent to the logger.

        Args:
            agent (object): Agent to save.
        """
        raise NotImplementedError("save_agent must be implemented in subclasses.")
    
    
@Logger.register('blank')
class BlankLogger(Logger):
    def close(self):
        """
        Performs any necessary logger cleanup.
        """
        pass

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        """
        Logs a dictionary of parameters. Parameters are values used to initialize but do not change throughout training.

        Args:
            params (dict): Dictionary of parameters.
        """
        pass

    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None,
                   ) -> None:
        """
        Logs a scalar value. Scalar values are any metric or value that changes throughout training.

        Args:
            name (str): Name of the scalar value.
            value (float): Value of the scalar value.
            iteration (int, optional): Iteration number.
        """
        pass

    def save_policy(self,
                    policy: BasePolicy,
                    ) -> None:
        """
        Saves the policy to the MLFlow run.

        Args:
            policy (BasePolicy): Policy to save.
        """
        pass

    def save_agent(self,
                    agent: object,
                    ) -> None:
        """
        Saves the agent to the MLFlow run.

        Args:
            agent (object): Agent to save.
        """
        pass

@Logger.register('file')
class FileLogger(Logger):
    def __init__(self,
                 output_dir: str
                 ) -> None:
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the output directory exists
        self.parameters = {}
        self.scalars = {}


    def close(self):
        """
        Writes the saved parameters and scalar metrics to a file.
        """
        param_file_path = os.path.join(self.output_dir, "parameters.json")
        with open(param_file_path, "w") as f:
            json.dump(self.parameters, f, indent=4)

        scalar_file_path = os.path.join(self.output_dir, "scalars.json")
        with open(scalar_file_path, "w") as f:
            json.dump(self.scalars, f, indent=4)

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        """
        Logs a dictionary of parameters.
        """
        self.parameters.update(params)


    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None,
                   ) -> None:
        """
        Logs scalar values, storing them sequentially or with a provided iteration number.
        """
        if name not in self.scalars:
            self.scalars[name] = []

        if iteration is None:
            iteration = len(self.scalars[name])

        self.scalars[name].append((iteration, value))

    def log_file(self,
                 path: str,
                 name: str,
                 move: bool = False
                 ) -> None:
        """
        Saves the given file to the output_dir/name folder.
        Creates the folder if it does not exist.
        """
        target_dir = os.path.join(self.output_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, os.path.basename(path))
        if move:
            shutil.move(path, target_path)
        else:
            shutil.copy(path, target_path)

    def save_policy(self,
                    policy,
                    name: str = "policy"
                    ) -> None:
        policy_path = os.path.join(self.output_dir, name)
        os.makedirs(policy_path, exist_ok=True)
        torch.save(policy, os.path.join(policy_path, "model.pth"))

@Logger.register('mlflow')
class MLFlowLogger(Logger):
    """
    MLFlow Logger

    Notes:
        psutil must be installed with pip to log system cpu metrics.
        pynvml must be installed with pip to log gpu metrics.

    References:
        [1] https://mlflow.org/docs/latest/python_api/mlflow.html
    """
    def __init__(self,
                 tracking_uri: str,
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 log_system_metrics: bool = False,
                 ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.iteration = 0

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(
            run_name=self.run_name,
            log_system_metrics=log_system_metrics,
        )

    def close(self):
        """
        Closes and cleans up the MLFlow logger.
        """
        mlflow.end_run()

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        """
        Logs a dictionary of parameters. Parameters are values used to initialize but do not change throughout training.
        Args:
            params (dict): Dictionary of parameters.
        """
        mlflow.log_params(params)

    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None
                   ) -> None:
        """
        Logs a scalar value to MLFlow.
        Args:
            name (str): Name of the scalar value.
            value (float): Value of the scalar value.
            iteration (int, optional): Iteration number.
        """
        mlflow.log_metric(name, value, step=iteration)

        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

    def save_agent(self,
                   agent: object,
                   agent_name: str = "agent.pt"
                   ) -> None:
        """
        Saves the agent to the MLFlow run.
        Args:
            agent (object): The agent object to save
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, agent_name)
            torch.save(agent, save_path)
            mlflow.log_artifact(save_path, artifact_path="agent")
        

    def save_policy(self,
                    policy: BasePolicy
                    ) -> None:

        """
        Saves the policy as a Python model so it can be registered in the MLFlow Registry.

        Args:
            policy (BasePolicy): The policy to be saved.
        """
        # Wrap policy in a PythonModel so it is a valid model
        class PolicyWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, policy: BasePolicy):
                self.policy = policy

            def predict(self, context, input_data):
                raise NotImplementedError('Policy loading is not implemented for RL policies.')

        # Save the policy type and dictionary representation to the model metadata
        policy_metadata = {
            'type': type(policy).__name__,
            'policy': policy.save_to_dict()
        }

        mlflow.pyfunc.log_model(
            artifact_path="policy",
            python_model=PolicyWrapper(policy),
            artifacts=None,
            conda_env=None,
            metadata=policy_metadata,
        )

