#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Generates the shock tube plots.
================================================================================
"""

from timeit import default_timer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pathlib

import shared_tools

plt.close('all')

# Global Variables
shock_tubes = ['b&w', 'd&w', 'rj1a', 'rj4d', 'einfeldt']
resolution = {'nx':512, 'ny':16, 'nz':16}
physical_size = 1.0

# Setup shock tube parameters
shock_tube_params = {}
coef = 1.0 / np.sqrt(4 * np.pi)
shock_tube_params['b&w']      = f'gamma=2.0 tout=0.1 outstep=0.1 diaph=0.5 '\
                                f'rho_l=1.0 vx_l=0 vy_l=0 vz_l=0 P_l=1.0 Bx_l=0.75 By_l=1.0 Bz_l=0.0 '\
                                f'rho_r=0.128 vx_r=0 vy_r=0 vz_r=0 P_r=0.1 Bx_r=0.75 By_r=-1.0 Bz_r=0.0'
shock_tube_params['d&w']      = f'gamma={5./3.} tout=0.2 outstep=0.2 diaph=0.5 '\
                                f'rho_l=1.08 vx_l=1.2 vy_l=0.01 vz_l=0.5 P_l=0.95 Bx_l={2.*coef} By_l={3.6*coef} Bz_l={2.*coef} '\
                                f'rho_r=1.0 vx_r=0 vy_r=0 vz_r=0 P_r=1.0 Bx_r={2.*coef} By_r={4.*coef} Bz_r={2.*coef}'
shock_tube_params['einfeldt'] = f'gamma=1.4 tout=0.16 outstep=0.16 diaph=0.5 '\
                                f'rho_l=1.0 vx_l=-2.0 vy_l=0 vz_l=0 P_l=0.45 Bx_l=0.0 By_l=0.5 Bz_l=0.0 '\
                                f'rho_r=1.0 vx_r=2.0 vy_r=0 vz_r=0 P_r=0.45 Bx_r=0.0 By_r=0.5 Bz_r=0.0'
shock_tube_params['rj1a']     = f'gamma={5./3.} tout=0.08 outstep=0.08 diaph=0.5 '\
                                f'rho_l=1.0 vx_l=10.0 vy_l=0 vz_l=0 P_l=20.0 Bx_l={5.0*coef} By_l={5.0*coef} Bz_l=0.0 '\
                                f'rho_r=1.0 vx_r=-10.0 vy_r=0 vz_r=0 P_r=1.0 Bx_r={5.0*coef} By_r={5.0*coef} Bz_r=0.0'
shock_tube_params['rj4d']     = f'gamma={5./3.} tout=0.16 outstep=0.16 diaph=0.5 '\
                                f'rho_l=1.0 vx_l=0 vy_l=0 vz_l=0 P_l=1.0 Bx_l=0.7 By_l=0.0 Bz_l=0.0 '\
                                f'rho_r=0.3 vx_r=0 vy_r=0 vz_r=1.0 P_r=0.2 Bx_r=0.7 By_r=1.0 Bz_r=0.0'

# ==============================================================================
def main():
    # Check for CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--in_path', help='The path to the directory that the source files are located in. Defaults to "~/Code/cholla/bin"')
    parser.add_argument('-o', '--out_path', help='The path of the directory to write the plots out to. Defaults to writing in the same directory as the input files')
    parser.add_argument('-r', '--run_cholla', action="store_true", help='Runs cholla to generate all the data')
    parser.add_argument('-f', '--figure', action="store_true", help='Generate the plots')
    parser.add_argument('-t', '--tube', default=['all'], nargs='+', help="List of tubes to run and/or plot. Options are 'b&w', 'd&w', 'rj1a', 'rj4d', 'einfeldt', and 'all'")

    args = parser.parse_args()

    if args.in_path:
        rootPath = pathlib.Path(str(args.in_path))
    else:
        rootPath = pathlib.Path(__file__).resolve().parent.parent

    if args.out_path:
        OutPath = pathlib.Path(str(args.out_path))
    else:
        OutPath = pathlib.Path(__file__).resolve().parent.parent / 'assets' / '3-mhd-tests'

    global shock_tubes
    if args.tube != ['all']:
        for test_val in args.tube:
            if test_val not in shock_tubes:
                raise ValueError(f'Unsupported value "{test_val}" given as a shock tube.')
        shock_tubes = args.tube

    if args.run_cholla:
        runCholla()

    if args.figure:
        plotShockTubes(rootPath, OutPath)
        for tube in shock_tubes:
            shared_tools.update_plot_entry(tube, 'python/shock-tubes.py')

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
                                   final_filename=f'{shock_tube}')

        # Print status
        print(f'Finished with {shock_tube}')
# ==============================================================================

# ==============================================================================
def plotShockTubes(rootPath, outPath):
    # Plotting info
    data_marker        = '.'
    data_markersize    = 5
    data_linestyle     = '-'
    linewidth          = 0.1 * data_markersize
    suptitle_font_size = 15
    subtitle_font_size = 10
    axslabel_font_size = 10
    tick_font_size     = 7.5

    # Field info
    fields = ['density', 'gas_pressure', 'energy', 'velocity_x', 'velocity_y', 'velocity_z',  'magnetic_x', 'magnetic_y', 'magnetic_z']
    field_indices = {'density':(0,0), 'gas_pressure':(0,1), 'energy':(0,2),
                     'velocity_x':(1,0), 'velocity_y':(1,1), 'velocity_z':(1,2),
                     'magnetic_x':(2,0), 'magnetic_y':(2,1), 'magnetic_z':(2,2)}

    # Plot the shock tubes data
    for shock_tube in shock_tubes:
        # Setup figure
        figSizeScale = 2.                 # Scaling factor for the figure size
        figHeight    = 4.8 * figSizeScale # height of the plot in inches, default is 4.8
        figWidth     = 7.0 * figSizeScale # width of the plot in inches, default is 6.4
        fig, subPlot = plt.subplots(3, 3, sharex=True, figsize = (figWidth, figHeight))

        # Whole plot settings
        fig.suptitle(f'{shared_tools.pretty_names[shock_tube]}', fontsize=suptitle_font_size)
        fig.tight_layout(pad = 1.5, w_pad = 1.5)

        # Load data
        data = shared_tools.load_conserved_data(f'{shock_tube}', load_gamma=True, load_resolution=True)
        data = shared_tools.center_magnetic_fields(data)
        data = shared_tools.slice_data(data,
                                       y_slice_loc=data['resolution'][1]//2,
                                       z_slice_loc=data['resolution'][2]//2)
        data = shared_tools.compute_velocities(data)
        data = shared_tools.compute_derived_quantities(data, data['gamma'])

        for field in fields:
            # Get info for this field
            subplot_idx = field_indices[field]
            field_data  = data[field]

            # Compute the positional data
            positions = np.linspace(0, physical_size, data[field].size)

            # Check the range. If it's just noise then set y-limits
            if np.abs(field_data.max() - field_data.min()) < 1E-10:
                mean = field_data.mean()
                subPlot[subplot_idx].set_ylim(mean - 0.5, mean + 0.5)

            # Plot the data
            subPlot[subplot_idx].plot(positions,
                                      field_data,
                                      color      = shared_tools.colors[field],
                                      linestyle  = data_linestyle,
                                      linewidth  = linewidth,
                                      marker     = data_marker,
                                      markersize = data_markersize,
                                      label      = 'PLMC')

            # Set ticks and grid
            subPlot[subplot_idx].tick_params(axis='both',
                                             direction='in',
                                             which='both',
                                             labelsize=tick_font_size,
                                             bottom=True,
                                             top=True,
                                             left=True,
                                             right=True)

            # Set titles
            subPlot[subplot_idx].set_ylabel(f'{shared_tools.pretty_names[field]}', fontsize=axslabel_font_size)
            if (subplot_idx[0] == 2):
                subPlot[subplot_idx].set_xlabel('Position', fontsize=axslabel_font_size)

        # Save the figure and close it
        plt.savefig(outPath / f'{shock_tube}.pdf', transparent = True)
        plt.close()

        print(f'Finished with {shared_tools.pretty_names[shock_tube]} plot.')
# ==============================================================================


if __name__ == '__main__':
    start = default_timer()
    main()
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')