import json
import mlflow
import mlflow.pyfunc
import numpy as np
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

    def __init__(self,
                 logging_freq: int = 1,
                 ) -> None:
        self.logging_freq = logging_freq
        self.last_logging_iteration = 0

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

    def should_log(self, iteration: int) -> bool:
        """
        Determines whether to log based on the current iteration and logging frequency.

        Args:
            iteration (int): Current iteration number.

        Returns:
            bool: True if logging should occur, False otherwise.
        """
        iteration += 1  # Adjust for 0-based indexing

        current_interval = iteration // self.logging_freq
        last_interval = self.last_logging_iteration // self.logging_freq
        if current_interval > last_interval:
            self.last_logging_iteration = iteration
            return True

        return False
    
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
                 output_dir: str,
                 logging_freq: int = 1,
                 ) -> None:
        super().__init__(logging_freq=logging_freq)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the output directory exists
        self.parameters = {}
        self.scalars = {}


    def close(self):
        """
        Writes the saved parameters and scalar metrics to a file.
        """
        def to_serializable(obj):
            if isinstance(obj, (np.generic, torch.Tensor)):
                return obj.item()
            return obj

        param_file_path = os.path.join(self.output_dir, "parameters.json")
        with open(param_file_path, "w") as f:
            json.dump(self.parameters, f, indent=4)

        scalar_file_path = os.path.join(self.output_dir, "scalars.json")
        with open(scalar_file_path, "w") as f:
            serializable_scalars = {
                k: [(int(step), to_serializable(value)) for step, value in v]
                for k, v in self.scalars.items()
            }
            json.dump(serializable_scalars, f, indent=4)

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

    def save_agent(self, 
                   agent: object,
                   name: str = "agent.pth"
                   ) -> None:
        """
        Saves the agent to the MLFlow run.
        Args:
            agent (object): The agent object to save
        """
        torch.save(agent, os.path.join(self.output_dir, name))

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
                 logging_freq: int = 1,
                 ) -> None:
        super().__init__(logging_freq=logging_freq)
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

@Logger.register('clearml')
class ClearMLLogger(Logger):
    """
    A lightweight adapter that logs training runs to **ClearML**.

    This logger mirrors the behavior of an MLflow-style logger while taking
    advantage of ClearML primitives:

    - Creates and manages a ClearML **Task** under the specified project.
    - Logs **parameters** as static configuration (via ClearML configuration /
      hyperparameters panels).
    - Logs **scalars** (metrics) with optional `"group/metric"` naming that maps
      to ClearML's *(title, series)* convention.
    - Saves an **agent** object as a versioned artifact attached to the Task.
    - Registers a **policy** as a first-class ClearML **Model** in the Model
      Registry (including design/metadata and uploaded weights), so it is
      discoverable and diffable alongside other models.

    The class expects ClearML to be installed and configured (e.g., via
    `clearml-init` or environment variables). It does **not** start or manage
    any background queues; uploads occur synchronously within each call.

    Attributes
    ----------
    project_name : str
        Name of the ClearML project under which the Task will be created.
    task_name : str
        Name of the ClearML Task (run) created for this logger instance.
    logging_freq : int
        Frequency hint inherited from :class:`Logger`. If your training loop
        calls `log_scalar` every step, you may use this value to conditionally
        log every *n*-th step (your loop is responsible for honoring it).
    iteration : int
        Monotonically increasing counter used as the default `iteration` (step)
        for scalar logging when none is supplied explicitly.
    task : clearml.Task
        The underlying ClearML Task object created during initialization.
    _logger : clearml.Logger
        The ClearML experiment logger obtained from `task.get_logger()`.

    Notes
    -----
    - **Scalar naming:** If you pass `"loss/train"` to `log_scalar`, it will be
      stored with `title="loss"` and `series="train"`. If no slash is present,
      the title defaults to `"metrics"` and the entire name becomes the series.
    - **Parameters:** Parameters are intended to be *static* (do not change over
      training). They are recorded into ClearML's configuration/hyperparameters
      view to support comparison and reproducibility.
    - **Models vs. Artifacts:** `save_policy` registers a ClearML *Model*
      (appears in the Model Registry) and uploads weights (e.g., a PyTorch
      `state_dict` or a pickle fallback). `save_agent` uploads a file or object
      as a Task *artifact* (appears under the run’s Artifacts panel).
    - **Environment:** Ensure ClearML is configured to point at your server
      (self-hosted or SaaS) via `clearml-init` or environment variables
      (`CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`).

    Examples
    --------
    Basic usage in a training loop:

    >>> logger = ClearMLLogger(project_name="Demo/PRT", task_name="PPO CartPole")
    >>> logger.log_parameters({"algo": "PPO", "seed": 42, "lr": 3e-4})
    >>> for step in range(1000):
    ...     loss = 1.0 / (step + 1)
    ...     reward = step * 0.1
    ...     logger.log_scalar("loss/train", loss)         # auto step (0,1,2,...)
    ...     logger.log_scalar("reward/mean", reward)
    >>> # Save objects
    >>> agent = {"type": "ppo", "notes": "demo"}
    >>> logger.save_agent(agent)                          # artifact on the Task
    >>> policy = my_policy                                # your BasePolicy impl
    >>> logger.save_policy(policy)                        # Model in Registry
    >>> logger.close()

    See Also
    --------
    clearml.Task : Underlying experiment/run object.
    clearml.OutputModel : Used for registering models in the ClearML registry.

    Raises
    ------
    ImportError
        If the `clearml` package is not installed or cannot be imported.
    """
    def __init__(self,
                 project_name: str,
                 task_name: str,
                 logging_freq: int = 1,
                 ) -> None:
        try:
            import clearml
            self.clearml = clearml
        except ImportError as e:
            raise ImportError("You need to install clearml to use this logger.") from e

        super().__init__(logging_freq=logging_freq)
        self.project_name = project_name
        self.task_name = task_name

        self.task = self.clearml.Task.init(project_name=self.project_name, task_name=self.task_name)
        self._logger = self.task.get_logger()

    def close(self):
        """
        Closes and cleans up the ClearML logger.
        """
        self.task.close()

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        """
        Logs a dictionary of parameters. Parameters are values used to initialize but do not change throughout training.

        Args:
            params (dict): Dictionary of parameters.
        """
        self.task.set_parameters_as_dict(params)

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
        # ClearML expects a (title, series) breakdown. Allow "group/metric" syntax; otherwise use a default group.
        if "/" in name:
            title, series = name.split("/", 1)
        else:
            title, series = "metrics", name

        step = self.iteration if iteration is None else iteration
        self._logger.report_scalar(title=title, series=series, value=value, iteration=step)

        # Keep our local iteration in sync like your MLFlowLogger
        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

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
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "agent.pt" 
            save_path = os.path.join(tmpdir, filename)
            torch.save(agent, save_path)

            # Upload file as an artifact (keeps original filename)
            self.task.upload_artifact(name="agent", artifact_object=save_path, metadata={"filename": filename})