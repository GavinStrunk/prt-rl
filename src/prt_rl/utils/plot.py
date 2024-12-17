import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from typing import Union, Dict


def plot_scalar_metric(
        metric_name: str,
        metric_values: Union[Dict, np.ndarray],
        max_value: float=None
) -> None:
    """
    Plots a confidence plot of the scalar metric.

    metric = {'metric_name':
    Support confidence max and min chopping
    Support color setting
    Support xlim and ylim setting
    Args:
        metric_name (str): name of the metric
        metric_values (dict): metric values
        max_value (float, optional): maximum value of the metric. Plots a horizontal line if maximum is provided. Defaults to None.

    Examples:

    """
    def _plot_metric(ax, label, data):
        if len(data.shape) == 1:
            ax.plot(data, label=label)
        elif len(data.shape) == 2:
            if data.shape[0] == 1:
                ax.plot(data[0], label=label)
            else:
                # Compute metric mean and standard deviation
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)

                # Compute confidence interval
                t_critical = t.ppf(0.975, df=data.shape[0] - 1)
                ci_margin = t_critical * (stds / np.sqrt(data.shape[0]))
                ci_upper = means + ci_margin
                ci_lower = means - ci_margin

                ax.plot(np.arange(data.shape[-1]), means, label=label)
                ax.fill_between(np.arange(data.shape[-1]), ci_lower, ci_upper, alpha=0.20)
        else:
            raise ValueError("Data must be 1D or 2D array")

    fig, ax = plt.subplots()

    # Handle if the metric values are provided as a dictionary of numpy arrays or just a single numpy array.
    if isinstance(metric_values, np.ndarray):
        _plot_metric(ax, "", metric_values)
    elif isinstance(metric_values, dict):
        for label, data in metric_values.items():
            _plot_metric(ax, label, data)
        plt.legend()
    else:
        raise ValueError("Metric values must be a dict of numpy arrays or a numpy array")

    # Plot horizontal line for maximum value
    if max_value is not None:
        ax.axhline(y=max_value, color='k', linestyle='--')

    # Add plot labels
    ax.set_title(metric_name)
    ax.set_xlabel('Episodes')
    ax.set_ylabel(metric_name)


