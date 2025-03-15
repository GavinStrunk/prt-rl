import mlflow
import mlflow.pyfunc
from typing import Optional
from prt_rl.common.policy import Policy


class Logger:
    """
    Based class for implementing loggers for RL algorithms.

    """
    def __init__(self):
        self.iteration = 0

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
                    policy: Policy,
                    ) -> None:
        """
        Saves the policy to the MLFlow run.

        Args:
            policy (Policy): Policy to save.
        """
        pass

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

        mlflow.set_tracking_uri(self.tracking_uri)
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

        mlflow.log_params(params)

    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None
                   ) -> None:
        mlflow.log_metric(name, value, step=iteration)

        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

    def save_policy(self,
                    policy: Policy
                    ) -> None:

        """
        Saves the policy as a Python model so it can be registered in the MLFlow Registry.

        Args:
            policy (Policy): The policy to be saved.
        """
        # Wrap policy in a PythonModel so it is a valid model
        class PolicyWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, policy: Policy):
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

