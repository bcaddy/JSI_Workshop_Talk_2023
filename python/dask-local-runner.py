#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 A simple skeleton for Dask scripts running on a single machine

 Dependencies:
     Dask
     timeit

================================================================================
"""

import dask
import dask.array as da
import dask.dataframe as dd
from dask import graph_manipulation
import argparse
import pathlib
import numpy as np

import cat_slice
import heatmap

# ==============================================================================
def main():
    # Get command line arguments
    cli = argparse.ArgumentParser()
    # Required Arguments
    cli.add_argument('-N', '--num-workers',    type=int,          required=True, help='The number of workers to use')
    # Optional Arguments
    cli.add_argument('--num-ranks',  type=int,  default=1,     help='The number of ranks cholla was run with')
    cli.add_argument('--cat-files',  type=bool, default=False, help='Concatenate the data files.')
    cli.add_argument('--gen-images', type=bool, default=False, help='Generate the images.')
    cli.add_argument('--gen-video',  type=bool, default=False, help='Convert the images to videos.')
    # none yet, feel free to add your own
    args = cli.parse_args()

    # Set scheduler type. Options are 'threads', 'processes', 'single-threaded', and 'distributed'.
    dask.config.set(scheduler='processes', num_workers=args.num_workers)
    # dask.config.set(scheduler='single-threaded')

    # Work to do
    num_outputs = 714
    outputs_to_work_on = np.arange(0, num_outputs+1)
    num_ranks = args.num_ranks

    root_directory        = pathlib.Path('/Users/bc9754/Scratch/orszag_tang_vortex/otv_small_2754x2754')
    source_directory      = root_directory / 'uncat_data'
    concat_file_directory = root_directory / 'data'
    image_file_directory  = root_directory / 'images'
    video_file_directory  = root_directory / 'videos'

    fields_to_skip = ['mz_xy', 'magnetic_z_xy'] # These fields have no evolution
    fields         = ['d_xy','mx_xy','my_xy','E_xy','magnetic_x_xy','magnetic_y_xy']

    fps = 24

    work_to_do = []
    for output in outputs_to_work_on:
        if args.cat_files:
            work_to_do.append(dask.delayed(cat_slice.concat_slice)(source_directory=source_directory,
                                                                destination_file_path=concat_file_directory / f'{output}_slice.h5',
                                                                num_ranks=num_ranks,
                                                                output_number=output,
                                                                concat_yz=False,
                                                                concat_xz=False,
                                                                skip_fields=fields_to_skip,
                                                                destination_dtype=np.float32))

        concat_idx = len(work_to_do)-1

        for field in fields:
            zoom = False
            if output == num_outputs:
                zoom = True
            image_task = dask.delayed(heatmap.generate_figure, pure=True)(concat_file_directory / f'{output}_slice.h5',
                                                               image_file_directory,
                                                               output,
                                                               field,
                                                               contour=False,
                                                               zoom=zoom,
                                                               fps=fps)
            if args.cat_files:
                image_task = dask.graph_manipulation.bind(image_task, work_to_do[concat_idx])
            if args.gen_images:
                work_to_do.append(image_task)

    pre_video_idx = len(work_to_do)
    for field in fields:
        video_task = dask.delayed(heatmap.make_video)(image_file_directory, video_file_directory, field, fps=fps)

        if args.cat_files or args.gen_images:
            video_task = dask.graph_manipulation.bind(video_task, work_to_do[:pre_video_idx])
        if args.gen_video:
            work_to_do.append(video_task)

    # Save the task graph
    print('starting visualize')
    dask.visualize(*work_to_do, filename=str(root_directory/'dask-task-graph.pdf'))

    # Execute the work
    print("starting compute")
    dask.compute(*work_to_do)
    print(f'Task complete: As requested finished work on {args.cat_files = }, {args.gen_images = }, and {args.gen_video = }')
# ==============================================================================

if __name__ == '__main__':
    from timeit import default_timer
    start = default_timer()
    main()
    print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
