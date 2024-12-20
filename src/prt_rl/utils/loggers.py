from abc import ABC, abstractmethod
import mlflow
from typing import Optional

class Logger(ABC):
    """
    Based class for implementing loggers for RL algorithms.

    """
    def __init__(self):
        self.iteration = 0

    @abstractmethod
    def log_scalar(self,
                   name: str,
                   value: float,
                   iteration: Optional[int] = None,
                   ) -> None:
        pass

class MLFlowLogger(Logger):
    """
    MLFlow Logger

    References:
        [1] https://mlflow.org/docs/latest/python_api/mlflow.html
    """
    def __init__(self, experiment_id, run_name, run_id, run_dir):
        super().__init__()
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.run_dir = run_dir

        mlflow.set_tracking_uri(f"{run_dir}")
        mlflow.set_experiment(self.experiment_id)
        mlflow.start_run()

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


