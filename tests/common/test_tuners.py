import pytest
import optuna
from prt_rl.common.tuners import OptunaTuner


def test_configure_int():
    study = optuna.create_study()
    trial = study.ask()

    param_dict = {
        'x': {
            'type': 'int',
            'low': 0,
            'high': 10
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'x' in params
    assert 0 <= params['x'] <= 10

    # Low must be > 0 to use log = True
    param_dict = {
        'y': {
            'type': 'int',
            'low': 1,
            'high': 100,
            'log': True
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'y' in params
    assert 1 <= params['y'] <= 100

    param_dict = {
        'z': {
            'type': 'int',
            'low': 0,
            'high': 100,
            'step': 5
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'z' in params
    assert 0 <= params['z'] <= 100

def test_invalid_configurations():
    study = optuna.create_study()
    trial = study.ask()

    param_dict = {
        'x': {
            'type': 'int',
            'low': 0,
            'high': 10,
            'log': True,
            'step': 2
        }
    }

    with pytest.raises(ValueError):
        OptunaTuner._configure_params(trial, param_dict)

        param_dict = {
        'x': {
            'type': 'float',
            'low': 0,
            'high': 10,
            'log': True,
            'step': 2
        }
    }

    with pytest.raises(ValueError):
        OptunaTuner._configure_params(trial, param_dict)

def test_configure_categorical():
    study = optuna.create_study()
    trial = study.ask()

    param_dict = {
        'x': {
            'type': 'categorical',
            'values': [8, 16, 32]
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'x' in params
    assert params['x'] in [8, 16, 32]

def test_configure_float():
    study = optuna.create_study()
    trial = study.ask()

    param_dict = {
        'x': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'x' in params
    assert 0.1 <= params['x'] <= 1.0

    param_dict = {
        'y': {
            'type': 'float',
            'low': 0.01,
            'high': 1.0,
            'log': True
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'y' in params
    assert 0.01 <= params['y'] <= 1.0

    param_dict = {
        'z': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0,
            'step': 0.1
        }
    }
    params = OptunaTuner._configure_params(trial, param_dict)
    assert 'z' in params

def test_optuna_tuner():
    def objective(params):
        x = params['x']
        return (x - 2) ** 2

    tuner = OptunaTuner(total_trials=100, maximize=True, num_jobs=1)
    best_params = tuner.tune(objective, parameters={'x': {'type':'float', 'low': -10, 'high': 10}})

    assert 'x' in best_params
    assert best_params['x'] == pytest.approx(-10, rel=0.1)