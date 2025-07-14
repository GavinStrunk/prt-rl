from abc import ABC, abstractmethod
import optuna
import numpy as np
from typing import Callable, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid
import multiprocessing
from torch import mul
from tqdm import tqdm

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
             objective_fcn: Callable,
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
    
class GridSearchTuner(HyperparameterTuner):
    """
    Hyperparameter tuning using Grid Search.

    Args:
        total_trials (int): The number of trials to run.
        maximize (bool): Whether to maximize the objective function. Default is True.
        num_jobs (int): The number of parallel jobs to run. Default is -1 (use all available cores).
    """

    def __init__(self, 
                 total_trials: int,
                 maximize: bool = True,
                 num_jobs: int = -1,
                 ) -> None:
        # Multiprocessing start method must be set to 'spawn' for compatibility with Pytorch's CUDA backend.
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                raise RuntimeError("Failed to set multiprocessing start method to 'spawn'. "
                                   "Ensure that this is called at the start of your script.")
            
        self.total_trials = total_trials
        self.maximize = maximize
        self.num_jobs = num_jobs if num_jobs != -1 else multiprocessing.cpu_count()

    def tune(self, 
             objective_fcn: Callable,
             parameters: dict,
             ) -> Dict[str, Any]:
        """
        Tune the hyperparameters of the given model using Grid Search.

        Args:
            objective_fcn (Callable[[Dict], float]): The objective function to be optimized.
            parameters (dict): The parameter dictionary that specifies the types and ranges to optimize.
        Returns:
            Dict[str, Any]: The best hyperparameters found during tuning.
        """
        grid = list(ParameterGrid(self._convert_parameters(parameters)))
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None

        # Use process-based parallelism
        with ProcessPoolExecutor(max_workers=self.num_jobs) as executor:
            futures = {executor.submit(objective_fcn, params): params for params in grid}

            with tqdm(total=len(futures), desc="Grid Search", unit="trial") as pbar:
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        score = future.result()
                        print(f"Trial finished - Params: {params}, Score: {score:.4f}")

                        is_better = (
                            (self.maximize and score > best_score) or
                            (not self.maximize and score < best_score)
                        )

                        if is_better:
                            best_score = score
                            best_params = params
                            print(f"🔥 New best trial! Score: {best_score:.4f}, Params: {best_params}")
                    except Exception as e:
                        print(f"❌ Trial failed for {params}: {e}")
                    finally:
                        pbar.update(1)

        return best_params        

    @staticmethod
    def _convert_parameters(parameters: dict) -> Dict[str, Any]:
        """
        Convert the parameter dictionary to a format suitable for grid search.
        
        Args:
            parameters (dict): The parameter dictionary.
        
        Returns:
            Dict[str, Any]: The converted parameter dictionary.
        """
        converted = {}
        for key, value in parameters.items():
            if value['type'] == 'categorical':
                converted[key] = value['values']
            elif value['type'] == 'float':
                if 'step' in value:
                    step = value['step']
                else:
                    step = (value['high'] - value['low']) / 10
                converted[key] = np.arange(value['low'], value['high'], step)
            elif value['type'] == 'int':
                if 'step' in value:
                    step = value['step']
                else:
                    step = 1
                converted[key] = np.arange(value['low'], value['high'], step)
            else:
                raise ValueError(f"Unsupported parameter type: {value['type']}")
        return converted