#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Generates the shock tube plots. Like the linear wave convergence plots this
 required that both a PLMC and PPMC version of the cholla executable are in the
 `bin` directory of the cholla submodule.
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

# matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "white"})

plt.close('all')

# Global Variables
shock_tubes = ['b&w']
resolution = {'nx':128, 'ny':16, 'nz':16}
physical_size = 1.0

# Setup shock tube parameters
shock_tube_params = {}
coef = 1.0 / np.sqrt(4 * np.pi)
shock_tube_params['b&w']      = f'gamma=2.0 tout=0.1 outstep=0.1 diaph=0.5 '\
                                f'rho_l=1.0 vx_l=0 vy_l=0 vz_l=0 P_l=1.0 Bx_l=0.75 By_l=1.0 Bz_l=0.0 '\
                                f'rho_r=0.128 vx_r=0 vy_r=0 vz_r=0 P_r=0.1 Bx_r=0.75 By_r=-1.0 Bz_r=0.0'

# ==============================================================================
def main():
    # Check for CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--in_path', help='The path to the directory that the source files are located in.')
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
        plotShockTubes(rootPath, OutPath, 'plmc')
        plotShockTubes(rootPath, OutPath, 'ppmc')


# ==============================================================================

# ==============================================================================
def runCholla():
    # Cholla settings
    common_settings = f"nx={resolution['nx']} ny={resolution['ny']} nz={resolution['nz']} init=Riemann " \
                      f"xmin=0.0 ymin=0.0 zmin=0.0 xlen={physical_size} ylen={physical_size} zlen={physical_size} "\
                       "xl_bcnd=3 xu_bcnd=3 yl_bcnd=3 yu_bcnd=3 zl_bcnd=3 zu_bcnd=3 "\
                       "outdir=./"

    # Loop over the lists and run cholla for each combination
    for shock_tube in shock_tubes:
        shared_tools.cholla_runner(cholla_cli_args=f'{common_settings} {shock_tube_params[shock_tube]}',
                                   move_final=True,
                                   reconstructor='plmc',
                                   final_filename=f'{shock_tube}_plmc')
        shared_tools.cholla_runner(cholla_cli_args=f'{common_settings} {shock_tube_params[shock_tube]}',
                                   move_final=True,
                                   reconstructor='ppmc',
                                   final_filename=f'{shock_tube}_ppmc')

        # Print status
        print(f'Finished with {shock_tube}')
# ==============================================================================

# ==============================================================================
def plotShockTubes(rootPath, outPath, reconstructor):
    # Plotting info
    data_marker        = '.'
    data_markersize    = 5
    data_linestyle     = '-'
    linewidth          = 0.1 * data_markersize
    title_font_size    = 35
    axslabel_font_size = 25
    tick_font_size     = 15
    axes_color         = '0.8'

    # Field info
    fields = ['velocity_x']

    # Plot the shock tubes data
    for shock_tube in shock_tubes:
        # Whole plot settings
        plt.title(f'{shared_tools.pretty_names[shock_tube]} {reconstructor.upper()}',
                  fontsize=title_font_size,
                  color=axes_color)

        # Load data
        data = shared_tools.load_conserved_data(f'{shock_tube}_{reconstructor}', load_gamma=True, load_resolution=True)
        # data = shared_tools.center_magnetic_fields(data)
        data = shared_tools.slice_data(data,
                                       y_slice_loc=data['resolution'][1]//2,
                                       z_slice_loc=data['resolution'][2]//2)
        data = shared_tools.compute_velocities(data)
        # data = shared_tools.compute_derived_quantities(data, data['gamma'])

        field_data  = data[fields[0]]

        # Compute the positional data
        positions = np.linspace(0, physical_size, data[fields[0]].size)

        # Plot the data
        plt.plot(positions,
                 field_data,
                 color      = shared_tools.colors[fields[0]],
                 linestyle  = data_linestyle,
                 linewidth  = linewidth,
                 marker     = data_marker,
                 markersize = data_markersize)

        # Set ticks and grid
        plt.tick_params(axis='both',
                        direction='in',
                        which='both',
                        labelsize=tick_font_size,
                        color=axes_color,
                        bottom=True,
                        top=True,
                        left=True,
                        right=True)

        # Set titles
        plt.ylabel(f'{shared_tools.pretty_names[fields[0]]}', fontsize=axslabel_font_size, color=axes_color)
        plt.xlabel('Position', fontsize=axslabel_font_size, color=axes_color)

        # Save the figure and close it
        plt.tight_layout()
        plt.savefig(outPath / f'{shock_tube}_{reconstructor}.pdf')#, transparent = True)
        plt.close()

        print(f'Finished with {shared_tools.pretty_names[shock_tube]} plot.')
# ==============================================================================


if __name__ == '__main__':
    start = default_timer()
    main()
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')