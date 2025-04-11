#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 Functions for generating Orszag-Tang Vortex plots and videos
================================================================================
"""
"""
TODO:
    - [x] videos of the 6 fields that 4,320 pixels high
    - [x] Label in corner with the name of the field
    - [x] images should fill field, axis in inside
    - [] different color maps for each field
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms

import numpy as np
import h5py

import os
import pathlib
import shutil

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

plt.rcParams['axes.labelsize']   = 25.0
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
def generate_figure(source_file_path, png_file_path, output_number, field, contour=False, zoom=False, pdf_file_path=None, fps=24):
    zoom=False
    # Some settings needed for the plot
    pretty_names = {'d_xy'         : "Density",
                    'mx_xy'        : "Momentum $X$",
                    'my_xy'        : "Momentum $Y$",
                    'E_xy'         : "Energy",
                    'magnetic_x_xy': "Magnetic Field $X$",
                    'magnetic_y_xy': "Magnetic Field $Y$"}

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
                 'magnetic_x_xy': 'PuBu',
                 'magnetic_y_xy': 'PuBu'}

    # Load the data
    with h5py.File(source_file_path, 'r') as data_file:
        data = np.empty(data_file[field].shape)
        data_file[field].read_direct(data)

    # Rotate the data to make it look like the standard plots
    data = np.rot90(data)

    # Loop for zooming
    final_size  = 40
    start_idx   = 0
    end_idx     = data.shape[0]
    zoom_frame  = 0
    while data.shape[0] > final_size:

        # subselect from the data
        if zoom and zoom_frame>0:
            zoom_step = int(np.ceil(data.shape[0]*0.0075/2))
            data = data[zoom_step:-zoom_step, zoom_step:-zoom_step]
            start_idx += zoom_step
            end_idx   -= zoom_step

        # Plotting
        plt.close('all')
        fig = plt.figure(figsize=(10,10))

        # Plot the main image
        extent = []
        extent.append(0) # x low limit
        extent.append(data.shape[0]) # x high limit
        extent.append(0) # y low limit
        extent.append(data.shape[1]) # y high limit
        plt.imshow(data,
                   cmap=color_map[field],
                   interpolation = 'none',
                   extent=extent,
                   vmin=low_limit[field],
                   vmax=high_limit[field],
                   origin='lower')

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

        num_ticks = 7
        labels    = np.linspace(start_idx, end_idx, num_ticks, dtype=int)
        locations = np.linspace(0, data.shape[0], num_ticks)
        plt.tick_params(axis="y", direction="in", pad=-28)
        plt.tick_params(axis="x", direction="in", pad=-30)
        plt.xticks(ticks=locations[1:], labels=labels[1:], rotation=-45, ha="right")
        plt.yticks(ticks=locations[1:], labels=labels[1:], rotation=-45, ha="right")

        # apply offset transform to all x ticklabels.
        offset = matplotlib.transforms.ScaledTranslation(-0.05, 0.0, fig.dpi_scale_trans)
        label = plt.gca().xaxis.get_majorticklabels()[-1]
        label.set_transform(label.get_transform() + offset)

        offset = matplotlib.transforms.ScaledTranslation(0.0, -0.08, fig.dpi_scale_trans)
        label = plt.gca().yaxis.get_majorticklabels()[-1]
        label.set_transform(label.get_transform() + offset)

        # Plot Settings, Titles, etc
        plt.text(2700, 2650, f"{pretty_names[field]}", size=20, ha='right')
        # plt.title(f'Exascale Orszag-Tang Vortex: {pretty_names[field]}')
        # plt.colorbar()
        # plt.xlabel(f'X-Direction Cells')
        # plt.ylabel(f'Y-Direction Cells')
        plt.tight_layout(pad=0.0)
        image_name = f'{field}_{int(output_number+zoom_frame)}'
        plt.savefig(f'{png_file_path}/{image_name}.svg')

        if zoom and zoom_frame == 0:
            for i in range(4*fps):
                zoom_frame += 1
                copy_image_name = f'{field}_{int(output_number+zoom_frame)}'
                shutil.copy(f'{png_file_path}/{image_name}.png', f'{png_file_path}/{copy_image_name}.png')

        plt.close()

        if not zoom:
            break

        zoom_frame += 1
# ======================================================================================================================

# ======================================================================================================================
def make_video(image_directory, video_directory, field, fps=24):
    # Convert the images to animations

    # 1080p is 1080, 2k is 1440, 4k is 2160
    # CURV display is 4,320 high and 19,200 wide
    resolution_height = 4320

    command = f"echo 'Y' | ffmpeg -r {fps} -i '{image_directory}/{field}_%d.png' -vf scale={resolution_height}:-2,setsar=1:1 -pix_fmt yuv420p {video_directory}/{field}.mp4 >/dev/null 2>&1"

    os.system(command)
# ======================================================================================================================
