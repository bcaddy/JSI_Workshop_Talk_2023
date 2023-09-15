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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
plt.rcParams['axes.titlesize']   = 18.0
plt.rcParams['figure.titlesize'] = 35.0

plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

# matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "black"})

# IPython_default = plt.rcParams.copy()
# print(IPython_default)

plt.close('all')
start = default_timer()

# ==============================================================================

# ==========================================================================
# Arguments
# ==========================================================================
# Check for CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--in_path', help='The path to the directory that the source files are located in. Defaults to "~/Code/cholla/bin"')
parser.add_argument('-o', '--out_path', help='The path of the directory to write the animation out to. Defaults to writing in the same directory as the input files')
parser.add_argument('-s', '--steps', default=1, help='How large should the step be between plots? Default is 1')
parser.add_argument('-d', '--dpi', default=300, help='The DPI of the images')

args = parser.parse_args()

if args.in_path:
    loadPath = pathlib.Path(str(args.in_path))
else:
    loadPath = pathlib.Path.home() / 'Code' / 'cholla' / 'bin'

if args.out_path:
    OutPath = pathlib.Path(str(args.out_path))
else:
    OutPath = loadPath

# Get the list of files
files = [f for f in os.listdir(loadPath) if os.path.isfile(os.path.join(loadPath, f))]

# Remove all non-HDF5 files and sort based on the time step
files = [f for f in files if ".h5" in f]
files.sort(key = lambda x: int(x.split(".")[0]))

# Get Attributes
file         = h5py.File(loadPath / '0.h5.0', 'r')
dims         = file.attrs['dims']
physicalSize = file.attrs['domain']
gamma        = file.attrs['gamma'][0]
numFiles     = len(files)

fields = [
          'density',
          'momentum_x',
          'momentum_y',
        #   'momentum_z',
          'Energy',
          'magnetic_x',
          'magnetic_y'
        #   'magnetic_z'
          ]

# Check that there are the right number of ranks
if len(fields) != comm.Get_size():
    raise Exception(f'Incorrect number of ranks. Expected {len(fields)} got {comm.Get_size()}.')

pretty_names = {'density':"Density", 'Energy':"Energy", 'momentum_x':"Momentum $x$",
                'momentum_y':"Momentum $y$", 'momentum_z':"Momentum $z$",
                'magnetic_x':"$B_x$", 'magnetic_y':"$B_y$", 'magnetic_z':"$B_z$"}

sliceLocation = dims[2] // 2

for field_rank,field in enumerate(fields):
    # If the field doesn't match this rank then skip it
    if field_rank != rank:
        continue

    # Load all the data
    data = []
    for i in np.arange(0, numFiles):
        dataFile = h5py.File(loadPath / f'{i}.h5.0', 'r')
        data.append(np.array(dataFile[field])[:, :, sliceLocation])
    data = np.array(data)


    # Plot the data
    low_lim  = np.min(data)
    high_lim = np.max(data)
    file_name_counter = 0

    if 'magnetic' in field:
        colormap = 'PuBu'
    else:
        colormap = 'inferno'

    for i in np.arange(0, numFiles, int(args.steps)):
        plt.figure(i)
        plt.imshow(data[i], cmap=colormap , interpolation = 'none', vmin=low_lim, vmax=high_lim, origin='lower')
        # plt.colorbar()
        plt.title(f'Orszag-Tang Vortex {pretty_names[field]}')
        # plt.xlabel(f'X-Direction Cells')
        # plt.ylabel(f'Y-Direction Cells')
        plt.tight_layout()
        plt.savefig(OutPath / 'images' / f'{field}-{file_name_counter}.png', dpi=int(args.dpi))
        plt.close()

        if field in ['density']:
            if i == 0 and field == 'density':
                data[0,0,0] += 1E-15
            plt.figure(i+numFiles)
            levels = np.linspace(np.min(data[i]), np.max(data[i]), 30)
            plt.contour(data[i], cmap=colormap, levels = levels)#, vmin=low_lim, vmax=high_lim)
            plt.gca().set_aspect('equal')
            # plt.colorbar()
            plt.title(f'Orszag-Tang Vortex {pretty_names[field]} Countour Plot')
            # plt.xlabel(f'X-Direction Cells')
            # plt.ylabel(f'Y-Direction Cells')
            plt.tight_layout()
            plt.savefig(OutPath / 'images' / f'{field}-contour-{file_name_counter}.png', dpi=int(args.dpi))
            plt.close()

        file_name_counter += 1

    # Convert the images to animations
    fps = np.max((file_name_counter // 10, 1))
    command = f"echo 'Y' | ffmpeg -r {fps} -i './images/{field}-%d.png' -pix_fmt yuv420p {field}.mp4 >/dev/null 2>&1"
    os.system(command)
    if field in ['density']:
        command = f"echo 'Y' | ffmpeg -r {fps} -i './images/density-contour-%d.png' -pix_fmt yuv420p density-contour.mp4 >/dev/null 2>&1"
        os.system(command)

    print(f'Finished with {field} data.')


comm.Barrier()
if rank == 0:
    print(f'Time to execute: {round(default_timer()-start,2)} seconds')
