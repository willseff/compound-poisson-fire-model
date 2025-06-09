import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_fire_size_distribution(data, fire_size_params,
    upper_limit=30000, 
    title='Fire Size Distribution',
    **kwargs):

    # Generate x values for plotting PDFs
    x = np.linspace(300, 30000, 1000)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    # Exponential
    axes[0].hist(data.clip(upper=upper_limit), bins=50, density=True, alpha=0.5, label='Observed', **kwargs)
    exp_scale = fire_size_params['exponential']
    axes[0].plot(x, stats.expon.pdf(x, scale=exp_scale), label='Exponential Fit', lw=2, color='C1')
    axes[0].set_title('Exponential Fit')
    axes[0].set_xlabel('Fire Size (ha)')
    axes[0].set_ylabel('Density')
    param_text = f"scale={exp_scale:.2f}"
    axes[0].text(0.05, 0.95, param_text, transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0].legend()
    axes[0].grid()

    # Weibull
    axes[1].hist(data.clip(upper=upper_limit), bins=50, density=True, alpha=0.5, label='Observed', **kwargs)
    weib_shape, weib_loc, weib_scale = fire_size_params['weibull']
    axes[1].plot(x, stats.weibull_min.pdf(x, weib_shape, loc=weib_loc, scale=weib_scale), label='Weibull Fit', lw=2, color='C2')
    axes[1].set_title('Weibull Fit')
    axes[1].set_xlabel('Fire Size (ha)')
    param_text = f"shape={weib_shape:.2f}\nloc={weib_loc:.2f}\nscale={weib_scale:.2f}"
    axes[1].text(0.05, 0.95, param_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].legend()
    axes[1].grid()

    # Pareto
    axes[2].hist(data.clip(upper=upper_limit), bins=50, density=True, alpha=0.5, label='Observed', **kwargs)
    pareto_b, pareto_loc, pareto_scale = fire_size_params['pareto']
    axes[2].plot(x, stats.pareto.pdf(x, pareto_b, loc=pareto_loc, scale=pareto_scale), label='Pareto Fit', lw=2, color='C3')
    axes[2].set_title('Pareto Fit')
    axes[2].set_xlabel('Fire Size (ha)')
    param_text = f"b={pareto_b:.2f}\nloc={pareto_loc:.2f}\nscale={pareto_scale:.2f}"
    axes[2].text(0.05, 0.95, param_text, transform=axes[2].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[2].legend()
    axes[2].grid()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_fire_count_distribution(data, fire_count_params, title, **kwargs):
    # same as above but there is two distributiions, exponential and gamma
    x = np.linspace(0, 120, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    # Exponential
    axes[0].hist(data, bins=15, density=True, alpha=0.5, label='Observed', **kwargs)
    exp_scale = fire_count_params['exponential']
    axes[0].plot(x, stats.expon.pdf(x, scale=exp_scale), label='Exponential Fit', lw=2, color='C1')
    axes[0].set_title('Exponential Fit')
    axes[0].set_xlabel('Fire Count')
    axes[0].set_ylabel('Density')
    param_text = f"scale={exp_scale:.2f}"
    axes[0].text(0.05, 0.95, param_text, transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0].legend()
    axes[0].grid()
    # Gamma
    axes[1].hist(data, bins=15, density=True, alpha=0.5, label='Observed', **kwargs)
    gamma_shape, gamma_loc, gamma_scale = fire_count_params['gamma']
    axes[1].plot(x, stats.gamma.pdf(x, gamma_shape, loc=gamma_loc, scale=gamma_scale), label='Gamma Fit', lw=2, color='C2')
    axes[1].set_title('Gamma Fit')
    axes[1].set_xlabel('Fire Count')
    param_text = f"shape={gamma_shape:.2f}\nloc={gamma_loc:.2f}\nscale={gamma_scale:.2f}"
    axes[1].text(0.05, 0.95, param_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].legend()
    axes[1].grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title)
    plt.show()


