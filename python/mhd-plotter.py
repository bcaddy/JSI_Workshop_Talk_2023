#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.  Created on Fri May 22 14:49:13 2020

 plot and animate results from a 1D MHD simulation. Based of example found
 [here](https://alexgude.com/blog/matplotlib-blitting-supernova/)

 Dependencies:
     numpy
     timeit
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

plt.rcParams['axes.labelsize']   = 20.0
plt.rcParams['figure.titlesize'] = 35.0

matplotlib.rcParams.update({"axes.grid" : True, "grid.color": "black"})

# IPython_default = plt.rcParams.copy()
# print(IPython_default)

plt.close('all')
start = default_timer()

# Setup the named tuples I'll need later
Fields = collections.namedtuple("Fields",("density", "pressure", "ie",
                                          "velocityX", "velocityY", "velocityZ",
                                          "magneticX", "magneticY", "magneticZ",
                                          "shape", "physicalSize", "timeStepNum", "time"),
                                defaults=[0,0,0,0])
Artists = collections.namedtuple("Artists",("density",   "pressure",  "ie",
                                            "velocityX", "velocityY", "velocityZ",
                                            "magneticX", "magneticY", "magneticZ",
                                            "titleArtist"))


# ==============================================================================
def main():
    # ==========================================================================
    # Settings
    # ==========================================================================
    # Check for CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fps_max',           help='Set the maximum frames per second')
    parser.add_argument('-p', '--in_path',           help='The path to the directory that the source files are located in. Defaults to "~/Code/cholla/bin"')
    parser.add_argument('-o', '--out_path',          help='The path of the directory to write the animation out to. Defaults to writing in the same directory as the input files')
    parser.add_argument('-n', '--out_name', default='out.mp4', help='The name of the output file, must end in ".mp4". Defaults to "out.mp4"')
    parser.add_argument('-a', '--axis', default='x', help='The axis along which to slice. Options are "x", "y", and "z"; defaults to "x".')

    args = parser.parse_args()

    if args.fps_max:
        fps = int(args.fps_max)
    else:
        fps = 20

    if args.in_path:
        loadPath = str(args.in_path)
    else:
        loadPath = str(pathlib.Path.home()) + '/Code/cholla/bin'

    if args.out_name:
        OutName = args.out_name

    if args.out_path:
        OutPath = str(args.out_path)
    else:
        OutPath = loadPath

    if args.axis == 'x':
        direction = 0
        xPosFrac = None
        yPosFrac = 0.5
        zPosFrac = 0.5
    elif args.axis == 'y':
        direction = 1
        xPosFrac = 0.5
        yPosFrac = None
        zPosFrac = 0.5
    elif args.axis == 'z':
        direction = 2
        xPosFrac = 0.5
        yPosFrac = 0.5
        zPosFrac = None
    else:
        print(f'Incorrect value of --axis passed in. Expected "x", "y", or "z", got {args.axis}')
        exit()

    # Plot Settings
    supTitleText  = "Time Evolution of Initial Conditions"
    # '#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2', '#fdb462', '#b3de69', '#bc82bd', '#ccebc4', '#ffed6f'
    colors = Fields(density  ='#ffed6f',  # color of the density plot
                    pressure ='#b3de69',  # color of the pressure plot
                    ie       ='#fdb462',  # color of the specific internal energy plot
                    velocityX='#fa8174',  # color of the velocity X plot
                    velocityY='#fa8174',  # color of the velocity Y plot
                    velocityZ='#fa8174',  # color of the velocity Z plot
                    magneticX='#81b1d2',  # color of the magnetic field X plot
                    magneticY='#81b1d2',  # color of the magnetic field Y plot
                    magneticZ='#81b1d2')  # color of the magnetic field Z plot
    linestyle     = '-'                # The line style
    linewidth     = 0.5                # How wide to make the lines
    marker        = "."                # Marker kind for points
    markersize    = 3                  # Size of the marker
    figSizeScale  = 2.                 # Scaling factor for the figure size
    figHeight     = 4.8 * figSizeScale # height of the plot in inches, default is 4.8
    figWidth      = 7.0 * figSizeScale # width of the plot in inches, default is 6.4
    padPercent    = 0.01               # How many percent larger the limits should be than the data

    # Set the coordinates for each figure
    coords = Fields(density   = (0,0), pressure  = (0,1),        ie = (0,2),
                    velocityX = (1,0), velocityY = (1,1), velocityZ = (1,2),
                    magneticX = (2,0), magneticY = (2,1), magneticZ = (2,2))

    # Set names of different plots
    prettyNames = Fields(density="Density", pressure="Pressure", ie="Internal Energy",
                         velocityX="$V_x$", velocityY="$V_y$", velocityZ="$V_z$",
                         magneticX="$B_x$", magneticY="$B_y$", magneticZ="$B_z$")

    # Video Settings
    duration      = 10.                         # How long the video is in seconds
    dpi           = 150                         # Dots per inch
    index         = 0                           # Initialize index
    initIndex     = 0                           # Index for init frames
    # fps is defined earlier in the command line settings section
    FrameTime     = (1./fps) * 1000             # Frametime in milliseconds
    # ==========================================================================
    # End settings
    # ==========================================================================

    # Load data
    data = loadData(loadPath, xPosFrac, yPosFrac, zPosFrac)

    # Compute which time steps to plot
    timeStepSamples, fps = selectTimeSteps(data.timeStepNum, duration, duration, fps)

    # Get the plot limits
    lowLimits, highLimits = computeLimits(data, padPercent)

    # Create the figure with 9 subplots
    fig, subPlot = plt.subplots(3, 3, figsize = (figWidth, figHeight))

    # Initialize the artists
    artists = createArtists(fig, subPlot, coords, prettyNames, linestyle, linewidth, marker, markersize, colors)

    # Setup Partial functions
    initPartial     = functools.partial(init, fig, subPlot, artists, coords, prettyNames, data, lowLimits, highLimits, supTitleText, direction)
    newFramePartial = functools.partial(newFrame, artists=artists, data=data, supTitleText=supTitleText, timeStepNum=data.timeStepNum, totFrames=timeStepSamples.size, direction=direction, time=data.time)

    # # Generate animation
    simulation = animation.FuncAnimation(fig=fig,
                                         func=newFramePartial,
                                         init_func=initPartial,
                                         frames = timeStepSamples,
                                         interval = FrameTime,
                                         blit = False,
                                         repeat = False)

    FFwriter = animation.FFMpegWriter(bitrate=1000,
                                      fps=fps,
                                      codec='libx264',
                                      extra_args=['-crf','28','-preset','ultrafast','-pix_fmt','yuv420p'])
    simulation.save(filename=f'{OutPath}/{OutName}', writer = FFwriter)
    # # simulation.save(filename=OutFile, fps=fps, dpi=dpi)
    print(f"\n\nAnimation complete. Framerate: {fps} fps, Total Number of Frames: {timeStepSamples.size}")
# ==============================================================================

# ==============================================================================
def loadData(path, xPosFrac, yPosFrac, zPosFrac):
    # Get the list of files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Remove all non-HDF5 files and sort based on the time step
    files = [f for f in files if ".h5" in f]
    files.sort(key = lambda x: int(x.split(".")[0]))

    # Get Attributes
    file         = h5py.File(path + "/0.h5.0", 'r')
    dims         = file.attrs['dims']
    physicalSize = file.attrs['domain']
    gamma        = file.attrs['gamma'][0]
    numFiles     = len(files)

    # Determine the positions to slice at
    if xPosFrac == None and yPosFrac != None and zPosFrac != None:
        direction = 0
        xMin = 0
        xMax = None
        yMin = int(dims[1] * yPosFrac)
        yMax = yMin + 1
        zMin = int(dims[2] * zPosFrac)
        zMax = zMin + 1
    elif xPosFrac != None and yPosFrac == None and zPosFrac != None:
        direction = 1
        xMin = int(dims[0] * xPosFrac)
        xMax = xMin + 1
        yMin = 0
        yMax = None
        zMin = int(dims[2] * zPosFrac)
        zMax = zMin + 1
    elif xPosFrac != None and yPosFrac != None and zPosFrac == None:
        direction = 2
        xMin = int(dims[0] * xPosFrac)
        xMax = xMin + 1
        yMin = int(dims[1] * yPosFrac)
        yMax = yMin + 1
        zMin = 0
        zMax = None
    else:
        print('No axis selected to slice on. Exiting.')
        exit()

    # Allocate Arrays
    densityData        = np.zeros((numFiles, dims[direction]))
    velocityXData      = np.zeros_like(densityData)
    velocityYData      = np.zeros_like(densityData)
    velocityZData      = np.zeros_like(densityData)
    pressureData       = np.zeros_like(densityData)
    ieData             = np.zeros_like(densityData)
    magneticXData      = np.zeros((numFiles, dims[direction]+1))
    magneticYData      = np.zeros_like(densityData)
    magneticZData      = np.zeros_like(densityData)
    timeStepNum        = np.zeros(numFiles)
    time               = np.zeros(numFiles)

    if direction == 0:
        magneticXData      = np.zeros((numFiles, dims[direction]+1))
        magneticYData      = np.zeros_like(densityData)
        magneticZData      = np.zeros_like(densityData)
    elif direction == 1:
        magneticXData      = np.zeros_like(densityData)
        magneticYData      = np.zeros((numFiles, dims[direction]+1))
        magneticZData      = np.zeros_like(densityData)
    elif direction == 2:
        magneticXData      = np.zeros_like(densityData)
        magneticYData      = np.zeros_like(densityData)
        magneticZData      = np.zeros((numFiles, dims[direction]+1))


    # Loop through files and load data
    for i, fileName in enumerate(files):
        file = h5py.File(path + '/' + fileName, 'r')

        timeStepNum[i] = file.attrs['n_step']
        time[i] = file.attrs['t']

        # Load data
        densityData[i, :]   = file['density'][xMin:xMax, yMin:yMax, zMin:zMax].flatten()
        velocityXData[i, :] = file['momentum_x'][xMin:xMax, yMin:yMax, zMin:zMax].flatten() / densityData[i, :]
        velocityYData[i, :] = file['momentum_y'][xMin:xMax, yMin:yMax, zMin:zMax].flatten() / densityData[i, :]
        velocityZData[i, :] = file['momentum_z'][xMin:xMax, yMin:yMax, zMin:zMax].flatten() / densityData[i, :]
        energyData          = file['Energy'][xMin:xMax, yMin:yMax, zMin:zMax].flatten()

        # Check for magnetic fields and import them if they exist
        if 'magnetic_x' in file:
            if xPosFrac == None:
                magneticXData[i, :] = file['magnetic_x'][:, yMin:yMax, zMin:zMax].flatten()
                magneticYData[i, :] = 0.5 * (file['magnetic_y'][:,  yMin:yMax, zMin:zMax] + file['magnetic_y'][:, yMin:yMax, zMin:zMax]).flatten()
                magneticZData[i, :] = 0.5 * (file['magnetic_z'][:,  yMin:yMax, zMin:zMax] + file['magnetic_z'][:, yMin:yMax, zMin+1:zMax+1]).flatten()
            elif yPosFrac == None:
                magneticXData[i, :] = 0.5 * (file['magnetic_x'][xMin:xMax, :,  zMin:zMax] + file['magnetic_x'][xMin+1:xMax+1, :, zMin:zMax]).flatten()
                magneticYData[i, :] = file['magnetic_y'][xMin:xMax, :, zMin:zMax].flatten()
                magneticZData[i, :] = 0.5 * (file['magnetic_z'][xMin:xMax, :,  zMin:zMax] + file['magnetic_z'][xMin:xMax, :, zMin+1:zMax+1]).flatten()
            elif zPosFrac == None:
                magneticXData[i, :] = 0.5 * (file['magnetic_x'][xMin:xMax, yMin:yMax, : ] + file['magnetic_x'][xMin+1:xMax+1, yMin:yMax, :]).flatten()
                magneticYData[i, :] = 0.5 * (file['magnetic_y'][xMin:xMax, yMin:yMax, : ] + file['magnetic_y'][xMin:xMax, yMin+1:yMax+1, :]).flatten()
                magneticZData[i, :] = file['magnetic_z'][xMin:xMax, yMin:yMax, :].flatten()

        # Compute more complex values
        velocitySquared = velocityXData[i, :]**2 + velocityYData[i, :]**2 + velocityZData[i, :]**2
        magneticSquared = (squareMagnetic(magneticXData[i, :],dims[direction]) +
                           squareMagnetic(magneticYData[i, :],dims[direction]) +
                           squareMagnetic(magneticZData[i, :],dims[direction]))

        # Compute pressures
        pressureData[i, :] = (gamma - 1) * (energyData
                                            - 0.5 * densityData[i, :] * (velocitySquared)
                                            - 0.5 * (magneticSquared))

        # Compute the specific internal energy
        # ieData[i, :] = energyData
        ieData[i, :] = (1/densityData[i, :]) * (energyData
                                                - 0.5 * densityData[i, :] * velocitySquared
                                                - 0.5 * magneticSquared)

    return Fields(density=densityData, pressure=pressureData, ie=ieData,
                  velocityX=velocityXData, velocityY=velocityYData, velocityZ=velocityZData,
                  magneticX=magneticXData, magneticY=magneticYData, magneticZ=magneticZData,
                  shape=dims, physicalSize=physicalSize, timeStepNum=timeStepNum, time=time)
# ==============================================================================

# ==============================================================================
def squareMagnetic(magnetic, targetSize):
    if magnetic.size == targetSize:
        return magnetic**2
    elif magnetic.size == targetSize+1:
        return (0.5 * (magnetic[:-1] + magnetic[1:]))**2
    else:
        raise ValueError("Magnetic field is not the right size for averaging")

# ==============================================================================

# ==============================================================================
def selectTimeSteps(timeStepNum, duration, totFrames, fps):
     # Compute which time steps to plot
    totFrames = fps*duration
    if timeStepNum.size >= totFrames:
        floatSamples    = np.arange(0, timeStepNum.size, timeStepNum.size/totFrames)
        timeStepSamples = np.asarray(np.floor(floatSamples), dtype="int")
    else:  # if the number of simulation steps is less than the total number of frames
        totFrames = timeStepNum.size
        fps       = np.ceil(totFrames/duration)
        FrameTime = (1./fps) * 1000
        timeStepSamples = np.arange(0, timeStepNum.size, 1, dtype="int")

    # Insert the initial second of the initial conditions
    return np.insert(timeStepSamples, 0, [0]*int(fps)), fps
# ==============================================================================

# ==============================================================================
def createArtists(fig, subPlot, coords, prettyNames, linestyle, linewidth, marker, markersize, color):
    outPlots = []

    # Generate Artists
    for i in range(len(coords)):
        # Skip elements set to default value
        if coords[i] != 0:
            returnPlot, = subPlot[coords[i]].plot([],
                                                  [],
                                                  linestyle  = linestyle,
                                                  linewidth  = linewidth,
                                                  marker     = marker,
                                                  markersize = markersize,
                                                  color      = color[i],
                                                  label      = prettyNames[i],
                                                  animated   = True)
            outPlots.append(returnPlot)

    return Artists(density   = outPlots[0], pressure = outPlots[1], ie        = outPlots[2],
                   velocityX = outPlots[3], velocityY= outPlots[4], velocityZ = outPlots[5],
                   magneticX = outPlots[6], magneticY= outPlots[7], magneticZ = outPlots[8],
                   titleArtist=fig.suptitle(''))
# ==============================================================================

# ==============================================================================
def computeLimit(dataSet, padPercent):
    pad = np.max(np.abs([np.nanmean(dataSet) - np.nanmin(dataSet), np.nanmax(dataSet)-np.nanmean(dataSet)])) * padPercent
    # pad = np.max(np.abs([np.nanmin(dataSet), np.nanmax(dataSet)])) * padPercent
    pad = pad if (pad > 0) else 0.5;
    pad = 0
    lowLim  = np.nanmin(dataSet) - pad
    highLim = np.nanmax(dataSet) + pad
    return lowLim, highLim
# ==============================================================================

# ==============================================================================
def computeLimits(data, padPercent):
    # Get the plot limits
    densityLowLim,   densityHighLim   = computeLimit(data.density,   padPercent)
    ieLowLim,        ieHighLim        = computeLimit(data.ie,        padPercent)
    pressureLowLim,  pressureHighLim  = computeLimit(data.pressure,  padPercent)
    velocityXLowLim, velocityXHighLim = computeLimit(data.velocityX, padPercent)
    velocityYLowLim, velocityYHighLim = computeLimit(data.velocityY, padPercent)
    velocityZLowLim, velocityZHighLim = computeLimit(data.velocityZ, padPercent)
    magneticXLowLim, magneticXHighLim = computeLimit(data.magneticX, padPercent)
    magneticYLowLim, magneticYHighLim = computeLimit(data.magneticY, padPercent)
    magneticZLowLim, magneticZHighLim = computeLimit(data.magneticZ, padPercent)

    # velocityXLowLim, velocityXHighLim = (1-1E-14, 1+1E-14)
    # velocityYLowLim, velocityYHighLim = (-1E-14,    1E-14)
    # velocityZLowLim, velocityZHighLim = (-1E-14,    1E-14)

    return (Fields(density=densityLowLim, pressure=pressureLowLim, ie=ieLowLim,
                   velocityX=velocityXLowLim, velocityY=velocityYLowLim, velocityZ=velocityZLowLim,
                   magneticX=magneticXLowLim, magneticY=magneticYLowLim, magneticZ=magneticZLowLim),
            Fields(density=densityHighLim, pressure=pressureHighLim, ie=ieHighLim,
                   velocityX=velocityXHighLim, velocityY=velocityYHighLim, velocityZ=velocityZHighLim,
                   magneticX=magneticXHighLim, magneticY=magneticYHighLim, magneticZ=magneticZHighLim))
# ==============================================================================

# ==============================================================================
def init(fig, subPlot, artists, coords, prettyNames, data, lowLimits, highLimits, supTitleText, direction):
    # Set up plots

    # Shared x-label
    subPlot[2,0].set_xlabel("Position")
    subPlot[2,1].set_xlabel("Position")
    subPlot[2,2].set_xlabel("Position")

    # Set title
    artists.titleArtist.set_text(f"{supTitleText} \n Time Step: 0, Time = 0")

    # Determine x limits
    xPad = 0.01
    xMin = 0 - xPad
    xMax = data.physicalSize[direction] + xPad

    # Set values for the subplots
    for i in range(len(prettyNames)):
        # Skip elements set to default value
        if prettyNames[i] != 0:
            subPlot[coords[i]].set_ylim(lowLimits[i], highLimits[i])
            subPlot[coords[i]].set_xlim(xMin, xMax)
            subPlot[coords[i]].set_ylabel(prettyNames[i])
            # subPlot[coords[i]].minorticks_on()  # Turning this on increases execution time by a factor of two
            subPlot[coords[i]].grid(which = 'both')

    # Layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    return artists
# ==============================================================================

# ==============================================================================
def newFrame(idx, artists, data, supTitleText, timeStepNum, totFrames, direction, time):
    artists.titleArtist.set_text(f"{supTitleText} \n Time Step: {int(timeStepNum[idx])}, Time = {time[idx]}")

    cellSize = data.physicalSize[direction] / data.shape[direction]
    cellCenteredPosition = np.linspace(cellSize/2, data.physicalSize[direction] - cellSize/2, data.shape[direction])
    faceCenteredPosition = np.linspace(0., data.physicalSize[direction], data.shape[direction]+1)

    artists.density  .set_data(cellCenteredPosition, data.density[idx,:])
    artists.pressure .set_data(cellCenteredPosition, data.pressure[idx,:])
    artists.ie       .set_data(cellCenteredPosition, data.ie[idx,:])

    artists.velocityX.set_data(cellCenteredPosition, data.velocityX[idx,:])
    artists.velocityY.set_data(cellCenteredPosition, data.velocityY[idx,:])
    artists.velocityZ.set_data(cellCenteredPosition, data.velocityZ[idx,:])

    if direction == 0:
        artists.magneticX.set_data(faceCenteredPosition, data.magneticX[idx,:])
        artists.magneticY.set_data(cellCenteredPosition, data.magneticY[idx,:])
        artists.magneticZ.set_data(cellCenteredPosition, data.magneticZ[idx,:])
    elif direction == 1:
        artists.magneticX.set_data(cellCenteredPosition, data.magneticX[idx,:])
        artists.magneticY.set_data(faceCenteredPosition, data.magneticY[idx,:])
        artists.magneticZ.set_data(cellCenteredPosition, data.magneticZ[idx,:])
    elif direction == 2:
        artists.magneticX.set_data(cellCenteredPosition, data.magneticX[idx,:])
        artists.magneticY.set_data(cellCenteredPosition, data.magneticY[idx,:])
        artists.magneticZ.set_data(faceCenteredPosition, data.magneticZ[idx,:])

    # Report progress
    if not hasattr(newFrame, "counter"):
        newFrame.counter = 0
        print()
    newFrame.counter += 1

    print(f'Animation is {100*(newFrame.counter/totFrames):.1f}% complete', end='\r')

    # The return is required to make blit work
    return artists
# ==============================================================================

main()

print(f'Time to execute: {round(default_timer()-start,2)} seconds')
