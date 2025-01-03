import mlflow
import mlflow.pyfunc
from typing import Optional
from prt_rl.utils.policies import Policy


class Logger:
    """
    Based class for implementing loggers for RL algorithms.

    """
    def __init__(self):
        self.iteration = 0

    def close(self):
        pass

    def log_parameters(self,
                       params: dict,
                       ) -> None:
        pass

    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None,
                   ) -> None:
        pass

    def save_policy(self,
                    policy: Policy,
                    ) -> None:
        pass

class MLFlowLogger(Logger):
    """
    MLFlow Logger

    Notes:
        psutil must be installed with pip to log system metrics.
        pynvml must be installed with pip to log gpu metrics.

    References:
        [1] https://mlflow.org/docs/latest/python_api/mlflow.html
    """
    def __init__(self,
                 tracking_uri: str,
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(
            run_name=self.run_name,
            log_system_metrics=True,
        )

    def close(self):
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

        class PolicyWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, policy: Policy):
                self.policy = policy


        mlflow.pyfunc.log_model(
            artifact_path="policy",
            python_model=PolicyWrapper(policy),
        )

