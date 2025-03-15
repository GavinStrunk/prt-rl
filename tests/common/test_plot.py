import matplotlib.pyplot as plt
import numpy as np
import pytest

from prt_rl.common.plot import plot_scalar_metric

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_single_scalar_metric():
    name = 'Random Value'
    value = np.random.rand(100)

    assert value.shape == (100,)
    plot_scalar_metric(name, value)
    plt.show()

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_single_scalar_with_one_agent():
    name = 'Random Value'
    value = np.random.rand(1, 100)

    assert value.shape == (1, 100)
    plot_scalar_metric(name, value)
    plt.show()

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_conf_interval_for_single_scalar():
    name = 'Random Value'
    value = np.random.rand(5, 100)

    assert value.shape == (5, 100)
    plot_scalar_metric(name, value)
    plt.show()

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_multiple_labels():
    name = 'Random Value'
    value1 = np.random.rand(100)
    value2 = np.random.rand(100)

    plot_scalar_metric(name, metric_values={'value1': value1, 'value2': value2})
    plt.show()

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_with_max_value():
    name = 'Random Value'
    value1 = np.random.rand(10, 100)
    value2 = np.random.rand(10, 100)

    plot_scalar_metric(name, metric_values={'value1': value1, 'value2': value2}, max_value=1.0)
    plt.show()

@pytest.mark.skip(reason="Generates a plot")
def test_plotting_invalid_scalar_data():
    name = 'Random Value'
    value = np.random.rand(1, 100, 2)

    assert value.shape == (1, 100, 2)
    with pytest.raises(ValueError):
        plot_scalar_metric(name, value)
