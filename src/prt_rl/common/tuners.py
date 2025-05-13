from abc import ABC, abstractmethod
import optuna
from typing import Callable, Dict, Any

class HyperparameterTuner(ABC):
    """
    Abstract base class for implementing hyperparameter tuners.
    """

    @abstractmethod
    def tune(self, 
             objective_fcn: Callable[[Dict], float],
             parameters: dict,
             ) -> Dict[str, Any]:
        """
        Tune the hyperparameters of the given objective function.
        Args:
            objective_fcn (Callable[[Dict], float]): The objective function to be optimized.
            parameters (dict): The parameter dictionary that specifies the types and ranges to optimize.
        Returns:
            Dict[str, Any]: The best hyperparameters found during tuning.
        """
        pass

class OptunaTuner(HyperparameterTuner):
    """
    Hyperparameter tuning using Optuna.

    Args:
        total_trials (int): The number of trials to run.
        maximize (bool): Whether to maximize the objective function. Default is True.
        num_jobs (int): The number of parallel jobs to run. Default is -1 (use all available cores).

    Example:
        .. python::
            param_dict = {
                'x': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 1.0
                },
                'y': {
                    'type': 'int',
                    'low': 1,
                    'high': 10,
                    'step': 1
                }
            }

            def objective(params):
                # Define your objective function here
                return (params['x'] - 2) ** 2 + (params['y'] - 5) ** 2

            tuner = OptunaTuner(total_trials=100)
            best_params = tuner.tune(objective, parameters=param_dict)
    """

    def __init__(self, 
                 total_trials: int,
                 maximize: bool = True,
                 num_jobs: int = -1,
                 ) -> None:
        self.total_trials = total_trials
        self.maximize = maximize
        self.num_jobs = num_jobs

    def tune(self, 
             objective_fcn: Callable[[Dict], float],
             parameters: dict,
             ) -> Dict[str, Any]:
        """
        Tune the hyperparameters of the given model using Optuna.

        Args:
            objective_fcn (Callable[[Dict], float]): The objective function to be optimized.
            parameters (dict): The parameter dictionary that specifies the types and ranges to optimize.
        Returns:
            Dict[str, Any]: The best hyperparameters found during tuning.
        """
        study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        study.optimize(lambda trial: OptunaTuner._objective(objective_fcn, trial, parameters),
                       catch=AssertionError,
                       show_progress_bar=True,
                       n_trials=self.total_trials,
                       n_jobs=self.num_jobs)
        
        trial = study.best_trial
        return trial.params
    
    @staticmethod
    def _objective(obj_fcn: Callable, trial: optuna.Trial, param_dict: dict) -> float:
        """
        Objective function for Optuna to optimize.
        Args:
            obj_fcn (callable): The objective function to be optimized.
            trial (optuna.Trial): The Optuna trial object.
            param_dict (dict): The parameter dictionary.
        Returns:
            The objective function value.
        """
        params = OptunaTuner._configure_params(trial, param_dict)
        return obj_fcn(params)

    @staticmethod
    def _configure_params(trial: optuna.Trial, param_dict: dict) -> dict:
        """
        Converts parameter dictionary definition to register values with the trial.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            param_dict (dict): Parameter dictionary that specifies the types and ranges to optimize

        Returns:
            A dictionary of parameter keys with the value for the current trial
        """
        params = {}
        for key, value in param_dict.items():
            val_type = value['type']
            if val_type == 'float':
                log_val = value['log'] if 'log' in value else False
                step = value['step'] if 'step' in value else None
                if log_val and step is not None:
                    raise ValueError("Log scale and step cannot be used together.")
                params[key] = trial.suggest_float(key, low=value['low'], high=value['high'], log=log_val, step=step)
            elif val_type == 'categorical':
                params[key] = trial.suggest_categorical(key, value['values'])
            elif val_type == 'int':
                log_val = value['log'] if 'log' in value else False
                step = value['step'] if 'step' in value else 1
                if log_val and step != 1:
                    raise ValueError("Log scale and step cannot be used together.")
                params[key] = trial.suggest_int(key, low=value['low'], high=value['high'], log=log_val, step=step)
            else:
                raise ValueError(f"Unsupported parameter type: {val_type}")

        return params