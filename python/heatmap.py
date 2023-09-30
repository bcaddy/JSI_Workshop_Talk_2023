#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Functions for generating Orszag-Tang Vortex plots and videos
================================================================================
"""

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import h5py

import os
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

plt.rcParams['axes.labelsize']   = 15.0
plt.rcParams['axes.titlesize']   = 35.0
plt.rcParams['figure.titlesize'] = 35.0

# plt.rcParams['xtick.bottom'] = False
# plt.rcParams['xtick.labelbottom'] = False
# plt.rcParams['ytick.left'] = False
# plt.rcParams['ytick.labelleft'] = False

# matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "black"})

# IPython_default = plt.rcParams.copy()
# print(IPython_default)

# ======================================================================================================================
def generate_figure(source_file_path, png_file_path, pdf_file_path, field, contour=False):
    # Some settings needed for the plot
    pretty_names = {'d_xy'         : "Density",
                    'mx_xy'        : "Momentum $x$",
                    'my_xy'        : "Momentum $y$",
                    'E_xy'         : "Energy",
                    'magnetic_x_xy': "$B_x$",
                    'magnetic_y_xy': "$B_y$"}

    low_limit =  {'d_xy'         :  0.00,
                  'mx_xy'        : -0.34,
                  'my_xy'        : -0.31,
                  'E_xy'         :  0.00,
                  'magnetic_x_xy': -0.74,
                  'magnetic_y_xy': -0.70}

    high_limit = {'d_xy'         : 0.50,
                  'mx_xy'        : 0.34,
                  'my_xy'        : 0.31,
                  'E_xy'         : 1.07,
                  'magnetic_x_xy': 0.74,
                  'magnetic_y_xy': 0.70}

    color_map = {'d_xy'         : 'inferno',
                 'mx_xy'        : 'inferno',
                 'my_xy'        : 'inferno',
                 'E_xy'         : 'inferno',
                 'magnetic_x_xy': 'inferno',
                 'magnetic_y_xy': 'inferno'}

    # Load the data
    with h5py.File(source_file_path, 'r') as data_file:
        data = np.empty(data_file[field].shape)
        data_file[field].read_direct(data)

    # Rotate the data to make it look like the standard plots
    data = np.rot90(data)

    # Plotting
    plt.figure(figsize=(10,10))

    # Plot the main image
    plt.imshow(data, cmap=color_map[field], interpolation = 'none', vmin=low_limit[field], vmax=high_limit[field], origin='lower')

    # If contour is turned on then overlay a contour plot
    if contour:
        if field == 'd_xy':
            # The density field at t=0 is constant so the contour plot errors.
            # Adding this tiny amount to the edge stops that error and isn't
            # visible in the final plot
            data[0,0] += 1E-15

        # Create evenly spaced levels
        levels = np.linspace(low_limit[field], high_limit[field], 30)

        plt.contour(data, cmap='Greys', levels = levels, vmin=low_limit[field], vmax=high_limit[field])
        plt.gca().set_aspect('equal')

    # Plot Settings, Titles, etc
    plt.title(f'Exascale Orszag-Tang Vortex: {pretty_names[field]}')
    # plt.colorbar()
    # plt.xlabel(f'X-Direction Cells')
    # plt.ylabel(f'Y-Direction Cells')
    plt.tight_layout()
    plt.savefig(f'{png_file_path}.png', dpi=400)
    plt.savefig(f'{pdf_file_path}.pdf')
    plt.close()
# ======================================================================================================================

# ======================================================================================================================
def make_video(image_directory, video_directory, field, fps=24):
    # Convert the images to animations

    # 1080p is 1080, 2k is 1440, 4k is 2160
    resolution_height = 2160

    command = f"echo 'Y' | ffmpeg -r {fps} -i '{image_directory}/{field}_%d.png' -vf scale={resolution_height}:-2,setsar=1:1 -pix_fmt yuv420p {video_directory}/{field}.mp4 >/dev/null 2>&1"

    os.system(command)
# ======================================================================================================================
