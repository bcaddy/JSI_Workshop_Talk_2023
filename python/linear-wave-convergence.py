#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Generates the linear wave convergence plots. This script it largely identical
 to the same script in the `analysis_scripts` repo, just optimized for this paper
================================================================================
"""

from timeit import default_timer
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import pathlib

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

# 1. (optionally) Run Cholla
#   a. Resolutions: 16, 32, 64, 128, 256, 512
#   b. All 4 waves
#   c. PLMC and PPMC
# 2. (optionally) Compute all the L2 Norms
# 3. (optionally) Plot the results
#   a. Plot specific scaling lines

# Lists to loop over
reconstructors = ['plmc', 'ppmc']
waves          = ['alfven_wave', 'fast_magnetosonic', 'mhd_contact_wave', 'slow_magnetosonic']
resolutions    = [16, 32, 64, 128, 256, 512]

# ==============================================================================
def main():
    # Check for CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--in_path', help='The path to the directory that the source files are located in. Defaults to "~/Code/cholla/bin"')
    parser.add_argument('-o', '--out_path', help='The path of the directory to write the plots out to. Defaults to writing in the same directory as the input files')
    parser.add_argument('-r', '--run_cholla', action="store_true", help='Runs cholla to generate all the data')
    parser.add_argument('-f', '--figure', action="store_true", help='Generate the plots')


    args = parser.parse_args()

    if args.in_path:
        rootPath = pathlib.Path(str(args.in_path))
    else:
        rootPath = pathlib.Path(__file__).resolve().parent.parent

    if args.out_path:
        OutPath = pathlib.Path(str(args.out_path))
    else:
        OutPath = pathlib.Path(__file__).resolve().parent.parent / 'assets'

    if args.run_cholla:
        runCholla()

    if args.figure:
        L2Norms = computeL2Norm(rootPath)
        plotL2Norm(L2Norms, OutPath)

# ==============================================================================

# ==============================================================================
def runCholla():
    # Loop over the lists and run cholla for each combination
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                shared_tools.cholla_runner(reconstructor=reconstructor,
                                           param_file_name=f'{wave}.txt',
                                           cholla_cli_args=f'nx={resolution} ny=16 nz=16',
                                           move_initial=True,
                                           move_final=True,
                                           initial_filename=f'{reconstructor}_{wave}_{resolution}_initial',
                                           final_filename=f'{reconstructor}_{wave}_{resolution}_final')

                # Print status
                print(f'Finished with {resolution}, {wave}, {reconstructor}')
# ==============================================================================

# ==============================================================================
def computeL2Norm(rootPath):
    # Setup dictionary to hold data
    l2_data = {}
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                l2_data[f'{reconstructor}_{wave}'] = []

    # Setup dictionary to hold data
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                # Determine file paths and load the files
                initial_data = shared_tools.load_conserved_data(f'{reconstructor}_{wave}_{resolution}_initial')
                final_data   = shared_tools.load_conserved_data(f'{reconstructor}_{wave}_{resolution}_final')

                # Get a list of all the data sets
                fields = initial_data.keys()

                # Compute the L2 Norm
                L2Norm = 0.0
                for field in fields:
                    diff = np.abs(initial_data[field] - final_data[field])
                    L1Error = np.sum(diff) / (initial_data[field]).size
                    L2Norm += np.power(L1Error, 2)

                L2Norm = np.sqrt(L2Norm)
                l2_data[f'{reconstructor}_{wave}'].append(L2Norm)

    return l2_data
# ==============================================================================

# ==============================================================================
def plotL2Norm(L2Norms, outPath, normalize = False):
    # Plotting info
    data_linestyle     = '-'
    linewidth          = 1
    plmc_marker        = '.'
    ppmc_marker        = '^'
    data_markersize    = 10
    scaling_linestyle  = '--'
    alpha              = 0.6
    scaling_color      = 'grey'
    title_font_size    = 30
    axslabel_font_size = 25
    legend_font_size   = 15
    tick_font_size     = 15

    for wave in waves:
        # Optionally, normalize the data
        if normalize:
            plmc_data = L2Norms[f'plmc_{wave}'] / L2Norms[f'plmc_{wave}'][0]
            ppmc_data = L2Norms[f'ppmc_{wave}'] / L2Norms[f'ppmc_{wave}'][0]
            norm_name = "Normalized "
        else:
            plmc_data = L2Norms[f'plmc_{wave}']
            ppmc_data = L2Norms[f'ppmc_{wave}']
            norm_name = ''

        # Plot raw data
        plt.plot(resolutions,
                                  plmc_data,
                                  color      = shared_tools.colors['plmc'],
                                  linestyle  = data_linestyle,
                                  linewidth  = linewidth,
                                  marker     = plmc_marker,
                                  markersize = data_markersize,
                                  label      = 'PLMC')
        plt.plot(resolutions,
                                  ppmc_data,
                                  color      = shared_tools.colors['ppmc'],
                                  linestyle  = data_linestyle,
                                  linewidth  = linewidth,
                                  marker     = ppmc_marker,
                                  markersize = 0.5*data_markersize,
                                  label      = 'PPMC')

        # Plot the scaling lines
        scalingRes = [resolutions[0], resolutions[1], resolutions[-1]]
        # loop through the different scaling powers
        for i in [2]:
            label = r'$\mathcal{O}(\Delta x^' + str(i) + r')$'
            norm_point = plmc_data[1]
            scaling_data = np.array([norm_point / np.power(scalingRes[0]/scalingRes[1], i), norm_point, norm_point / np.power(scalingRes[-1]/scalingRes[1], i)])
            plt.plot(scalingRes, scaling_data, color=scaling_color, alpha=alpha, linestyle=scaling_linestyle, linewidth=linewidth, label=label)

        # Set axis parameters
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1E1, 1E3)

        # Set ticks and grid
        plt.tick_params(axis='both', direction='in', which='both', labelsize=tick_font_size, bottom=True, top=True, left=True, right=True)

        # Set axis titles
        plt.xlabel('Resolution', fontsize=axslabel_font_size)
        plt.ylabel(f'{norm_name}L2 Error', fontsize=axslabel_font_size)
        plt.title(f'{shared_tools.pretty_names[wave]}', fontsize=title_font_size)

        plt.legend(fontsize=legend_font_size)
        plt.grid(color='0.25')

        plt.tight_layout()
        plt.savefig(outPath / f'linear_convergence_{wave}.pdf')
        plt.close()
# ==============================================================================


if __name__ == '__main__':
    start = default_timer()
    main()
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')