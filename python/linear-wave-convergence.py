#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.  Created on %(date)s

 Description (in paragraph form)

 Dependencies:
     numpy
     timeit
     donemusic
     matplotlib

 Changelog:
     Version 1.0 - First Version
================================================================================
"""

from timeit import default_timer

import collections
import functools

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import h5py

import os
import sys
import argparse
import pathlib

matplotlib.use("Agg")
plt.style.use('dark_background')

background_color = '0.1'
plt.rcParams['axes.facecolor']    = background_color
plt.rcParams['figure.facecolor']  = background_color
plt.rcParams['patch.facecolor']   = background_color
plt.rcParams['savefig.facecolor'] = background_color

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

plt.rcParams['axes.labelsize']   = 18.0
plt.rcParams['axes.titlesize']   = 20.0
plt.rcParams['figure.titlesize'] = 35.0

# matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "black"})

# IPython_default = plt.rcParams.copy()
# print(IPython_default)

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
    parser.add_argument('-r', '--run_cholla', default=False, help='Runs cholla to generate all the scaling data')
    parser.add_argument('-f', '--figure', default=False, help='Plot the L2 Norms')


    args = parser.parse_args()

    if args.in_path:
        rootPath = pathlib.Path(str(args.in_path))
    else:
        rootPath = pathlib.Path.home() / 'Code' / 'cholla' / 'bin'

    if args.out_path:
        OutPath = pathlib.Path(str(args.out_path))
    else:
        OutPath = rootPath

    if args.run_cholla == 'True':
        runCholla(rootPath)

    if args.figure == 'True':
        L2Norms = computeL2Norm(rootPath)
        plotL2Norm(L2Norms, OutPath)
        plotL2Norm(L2Norms, OutPath, True)
# ==============================================================================

# ==============================================================================
def runCholla(rootPath):
    # Basic Settings
    offAxisResolution = 'ny=16 nz=16'

    # Loop over the lists and run cholla for each combination
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                # Generate Cholla run command
                chollaPath = rootPath / f'cholla.mhd.c3po.{reconstructor}'
                paramFilePath = rootPath / f'{wave}.txt'
                logFile = rootPath / 'cholla.log'
                command = f'{chollaPath} {paramFilePath} nx={resolution} {offAxisResolution} >> {logFile} 2>&1'

                # Run Cholla
                os.system(command)

                # Move data files
                initialFile = rootPath / '0.h5.0'
                finalFile = rootPath / '1.h5.0'
                initialFile.rename(rootPath / 'data' / f'{reconstructor}_{wave}_{resolution}_initial.h5')
                finalFile.rename(rootPath / 'data' / f'{reconstructor}_{wave}_{resolution}_final.h5')

                # Print status
                print(f'Finished with {resolution}, {wave}, {reconstructor}')
# ==============================================================================

# ==============================================================================
def computeL2Norm(rootPath):
    # Setup dictionary to hold data
    data = {}
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                data[f'{reconstructor}_{wave}'] = []

    # Loop over the lists and compute the L2 norm for each combination
    for reconstructor in reconstructors:
        for wave in waves:
            for resolution in resolutions:
                # Determine file paths and load the files
                initialFilePath = rootPath / 'data' / f'{reconstructor}_{wave}_{resolution}_initial.h5'
                finalFilePath   = rootPath / 'data' / f'{reconstructor}_{wave}_{resolution}_final.h5'
                initialFile = h5py.File(initialFilePath, 'r')
                finalFile   = h5py.File(finalFilePath, 'r')

                # Get a list of all the data sets
                fields = initialFile.keys()

                # Compute the L2 Norm
                L2Norm = 0.0
                for field in fields:
                    initialData = np.array(initialFile[field])
                    finalData   = np.array(finalFile[field])

                    diff = np.abs(initialData - finalData)
                    L1Error = np.sum(diff) / initialData.size
                    L2Norm += np.power(L1Error, 2)

                L2Norm = np.sqrt(L2Norm)
                data[f'{reconstructor}_{wave}'].append(L2Norm)

    return data
# ==============================================================================

# ==============================================================================
def plotL2Norm(L2Norms, outPath, normalize = False):
    # Pretty names
    pretty_names = {'alfven_wave':r'Alfven Wave',
                    'fast_magnetosonic':'Fast Magnetosonic Wave',
                    'mhd_contact_wave':'Entropy Wave',
                    'slow_magnetosonic':'Slow Magnetosonic Wave'}

    # Plotting info
    data_linestyle  = '-'
    linewidth  = 1
    data_marker     = '.'
    data_markersize = 10
    scaling_linestyle = '--'
    alpha = 0.6
    scaling_color = 'white'
    annotate_font_size = 13

    # Plot the L2 Norm data
    for wave in waves:
        plt.figure()
        ax  = plt.gca()

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
        plt.plot(resolutions, plmc_data, color='#8dd3c7', linestyle=data_linestyle, linewidth=linewidth, marker=data_marker, markersize=data_markersize, label='PLMC')
        plt.plot(resolutions, ppmc_data, color='#feffb3', linestyle=data_linestyle, linewidth=linewidth, marker=data_marker, markersize=data_markersize, label='PPMC')

        # Plot the scaling lines
        scalingRes = [resolutions[0], resolutions[1], resolutions[-1]]
        # loop through the different scaling powers
        for i in [2]:
            norm_point = plmc_data[1]
            scaling_data = np.array([norm_point / np.power(scalingRes[0]/scalingRes[1], i), norm_point, norm_point / np.power(scalingRes[-1]/scalingRes[1], i)])
            plt.plot(scalingRes, scaling_data, color=scaling_color, alpha=alpha, linestyle=scaling_linestyle, linewidth=linewidth)
            label = r'$\mathcal{O}(\Delta x^' + str(i) + r')$'
            plt.annotate(label, xy=(scalingRes[-1], scaling_data[-1]), fontsize=annotate_font_size, textcoords='offset points', xytext=(2, -8))

        # Set axis parameters
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1E1, 1E3)

        # Legend
        plt.legend()

        # Set ticks
        ax.tick_params(axis='both', direction='in', which='both', labelsize=annotate_font_size, bottom=True, top=True, left=True, right=True)

        # Set labels
        plt.title(f'{norm_name}MHD Linear Wave Convergence\n({pretty_names[wave]})')
        plt.xlabel('Resolution')
        plt.ylabel(f'{norm_name}L2 Error')

        plt.tight_layout()
        plt.savefig(outPath / f'{wave}_linear_convergence{norm_name}.pdf')
        plt.close()
# ==============================================================================


if __name__ == '__main__':
    start = default_timer()
    main()
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')