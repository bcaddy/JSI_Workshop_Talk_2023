#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Generates the scaling test plots. This script it largely identical
 to the same script in the `scaling-tests` repo, just optimized for this paper
================================================================================
"""

from timeit import default_timer
import numpy as np
import pandas as pd
import pathlib
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import shared_tools

matplotlib.use("Agg")
plt.style.use('dark_background')

axes_color = '0.1'
# plt.rcParams['axes.facecolor']    = axes_color
# plt.rcParams['figure.facecolor']  = background_color
# plt.rcParams['patch.facecolor']   = background_color
# plt.rcParams['savefig.facecolor'] = background_color

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

plt.close('all')

# ==============================================================================
def main():
    data_path = shared_tools.repo_root / 'scaling-tests' / 'data' / '2023-08-24-plmc'

    scaling_data = load_data(data_path)

    x_limit = [0.7, 1E5]

    Scaling_Plot(scaling_data=scaling_data,
                 y_title=r'Cells / Second / GPU',
                 filename='cells_per_second',
                 plot_func=cells_per_second_per_gpu,
                 xlims=x_limit,
                 skip_mpi=True)

#     Scaling_Plot(scaling_data=scaling_data,
#                  y_title=r'Milliseconds / 256$^3$ Cells / GPU',
#                  filename='ms_per_gpu',
#                  plot_func=ms_per_256_per_gpu,
#                  xlims=x_limit)

#     Scaling_Plot(scaling_data=scaling_data,
#                  y_title=r'Weak Scaling Efficiency',
#                  filename='weak_scaling_efficiency',
#                  plot_func=weak_scaling_efficiency,
#                  xlims=x_limit,
#                  skip_mpi=True,
#                  skip_integrator=True,
#                  legend=False)
# # ==============================================================================

# ==============================================================================
def load_data(data_path):
    # Get paths
    data_dirs = sorted(pathlib.Path(data_path).glob('ranks*'))

    # Setup dataframe
    file_name = data_dirs[0] / 'run_timing.log'
    with open(file_name, 'r') as file:
        lines        = file.readlines()
        header       = lines[3][1:].split()
        scaling_data = pd.DataFrame(index=header)

    # Loop through the directories and load the data
    for path in data_dirs:
        file_name = path / 'run_timing.log'
        if file_name.is_file():
            with open(file_name, 'r') as file:
                lines  = file.readlines()
                data   = lines[4].split()
                scaling_data[int(data[0])] = pd.to_numeric(data)
        else:
            print(f'File: {file_name} not found.')

    scaling_data = scaling_data.reindex(sorted(scaling_data.columns), axis=1)
    return scaling_data
# ==============================================================================

# ==============================================================================
def Scaling_Plot(scaling_data, y_title, filename, plot_func, xlims, ylims=None, skip_mpi=False, skip_integrator=False, legend=True):
    # Instantiate Plot
    fig = plt.figure(0, figsize=(15, 10))
    fig.clf()
    ax = plt.gca()

    # Set plot settings
    color_mhd       = 'blue'
    color_mpi       = 'red'
    color_total     = 'purple'
    marker_size     = 10

    # Plot the data
    ax = plot_func(scaling_data, 'Total', color_total, 'Total', ax, marker_size)
    if (not skip_integrator):
        # Note that the timer for the integrator is named "Hydro" not "MHD"
        ax = plot_func(scaling_data, 'Hydro_Integrator', color_mhd, 'MHD Integrator', ax, marker_size)
    if (not skip_mpi):
        ax = plot_func(scaling_data, 'Boundaries', color_mpi, 'MPI Communication', ax, marker_size, delete_first=True)

    # Set CPU region
    num_procs = scaling_data.loc['n_proc'].to_numpy()
    ax.text(1.02E2, 2E6,
            '   Typical CPU Code\nCells / Second / CPU',
            fontsize=20,
            alpha=0.5)
    athenaPP_per_cpu = 20 * 1E6
    cpu_high_lim = 2. * athenaPP_per_cpu
    ax.fill_between(x=[num_procs.min(), num_procs.max()], y1=[1E6, 1E6], y2=[cpu_high_lim, cpu_high_lim],
                    color='cyan',
                    alpha=0.2)

    # Setup the rest of the plot
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ymin = 1, ymax=1E9)

    ax.set_yscale('log')
    ax.set_xscale('log')

    locmaj = matplotlib.ticker.LogLocator(base=10.0,
                                          subs=(1.0, ),
                                          numticks=100)
    ax.yaxis.set_major_locator(locmaj)

    locmin = matplotlib.ticker.LogLocator(base=10.0,
                                          subs=np.arange(2, 10) * .1,
                                          numticks=100)
    ax.yaxis.set_minor_locator(locmin)
    ax.tick_params(which='both', direction='in', labelsize=20, bottom=True, top=True, left=True, right=True)

    plt.grid(axis='x', color='0.5', which='major')
    plt.grid(axis='y', color='0.5', which='major')
    # plt.grid(axis='y', color='0.25', which='minor')


    title_size      = 40
    axis_label_size = 25
    fig.suptitle('MHD Weak Scaling on Frontier (PLMC)', fontsize=title_size)
    ax.set_ylabel(y_title, fontsize=axis_label_size)
    ax.set_xlabel(r'Number of GPUs', fontsize=axis_label_size)

    if legend:
        legend = ax.legend(fontsize=15)
    fig.tight_layout()

    output_path = shared_tools.repo_root / 'assets' / f'scaling_tests_{filename}.pdf'
    fig.savefig(output_path, dpi=400)
# ==============================================================================

# ==============================================================================
def ms_per_256_per_gpu(scaling_data, name, color, label, ax, marker_size, delete_first=False):
    x = scaling_data.loc['n_proc'].to_numpy()
    y = scaling_data.loc[name].to_numpy() / (scaling_data.loc['n_steps'].to_numpy() - 1)

    if (delete_first):
        x = x[1:]
        y = y[1:]

    ax.plot(x, y, '--', c=color, marker='o', markersize=marker_size, label=label)

    return ax
# ==============================================================================

# ==============================================================================
def weak_scaling_efficiency(scaling_data, name, color, label, ax, marker_size, delete_first=None):
    x = scaling_data.loc['n_proc'].to_numpy()
    y = scaling_data.loc[name].to_numpy() / (scaling_data.loc['n_steps'].to_numpy() - 1)

    # Normalize
    y = y[0] / y

    ax.plot(x, y, '--', c=color, marker='o', markersize=marker_size)

    return ax
# ==============================================================================

# ==============================================================================
def cells_per_second_per_gpu(scaling_data, name, color, label, ax, marker_size, delete_first=None):
    x = (scaling_data.loc['n_proc'].to_numpy())

    cells_per_gpu = ( scaling_data.loc['nx'].to_numpy()
                    * scaling_data.loc['ny'].to_numpy()
                    * scaling_data.loc['nz'].to_numpy()) / x

    avg_time_step = scaling_data.loc[name].to_numpy() / (scaling_data.loc['n_steps'].to_numpy() - 1)
    avg_time_step /= 1000 # convert to seconds from ms

    y = cells_per_gpu / avg_time_step

    ax.plot(x, y, '--', c=color, marker='o', markersize=marker_size, label=label)

    return ax
# ==============================================================================

if __name__ == '__main__':
    start = default_timer()
    main()
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')
