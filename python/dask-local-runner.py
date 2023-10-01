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
    cli = argparse.ArgumentParser()
    # Required Arguments
    cli.add_argument('-n', '--num-workers', type=int, default=6, help='The number of workers to use.')
    # Optional Arguments
    cli.add_argument('--cat-files',  type=bool, default=False, help='Concatenate the data files.')
    cli.add_argument('--gen-images', type=bool, default=False, help='Generate the images.')
    cli.add_argument('--gen-video',  type=bool, default=False, help='Convert the images to videos.')
    args = cli.parse_args()

    # Set scheduler type. Options are 'threads', 'processes', 'single-threaded', and 'distributed'.
    dask.config.set(scheduler='processes', num_workers=args.num_workers)
    # dask.config.set(scheduler='single-threaded')

    # Work to do
    outputs_to_work_on = [0, 1, 714]
    num_ranks = 16

    root_directory        = pathlib.Path.home() / 'Downloads' / 'small_otv_test_data'
    source_directory      = root_directory / 'uncat_data'
    concat_file_directory = root_directory / 'outdir'
    png_file_directory    = root_directory / 'images' / 'png'
    pdf_file_directory    = root_directory / 'images' / 'pdf'
    video_file_directory  = root_directory / 'videos'

    fields_to_skip = ['mz_xy', 'magnetic_z_xy'] # These fields have no evolution
    fields         = ['d_xy','mx_xy','my_xy','E_xy','magnetic_x_xy','magnetic_y_xy']

    work_to_do = []
    for output in outputs_to_work_on:
        if args.cat_files:
            work_to_do.append(dask.delayed(cat_slice.concat_slice)(source_directory=source_directory,
                                                                destination_file_path=concat_file_directory / f'{output}_slice.h5',
                                                                num_ranks=num_ranks,
                                                                timestep_number=output,
                                                                concat_yz=False,
                                                                concat_xz=False,
                                                                skip_fields=fields_to_skip,
                                                                destination_dtype=np.float32))

        concat_idx = len(work_to_do)-1

        for field in fields:
            image_name = f'{field}_{output}'
            image_task = dask.delayed(heatmap.generate_figure)(concat_file_directory / f'{output}_slice.h5',
                                                               png_file_directory / image_name,
                                                               pdf_file_directory / image_name,
                                                               field,
                                                               contour=False)
            if args.cat_files:
                image_task = dask.graph_manipulation.bind(image_task, work_to_do[concat_idx])
            if args.gen_images:
                work_to_do.append(image_task)

    pre_video_idx = len(work_to_do)
    for field in fields:
        video_task = dask.delayed(heatmap.make_video)(png_file_directory, video_file_directory, field, fps=1)

        if args.cat_files or args.gen_images:
            video_task = dask.graph_manipulation.bind(video_task, work_to_do[:pre_video_idx])
        if args.gen_video:
            work_to_do.append(video_task)

    # Save the task graph
    dask.visualize(*work_to_do, filename=str(pathlib.Path(__file__).resolve().parent/'dask-task-graph.pdf'))

    # Execute the work
    dask.compute(*work_to_do)
# ==============================================================================

if __name__ == '__main__':
    from timeit import default_timer
    start = default_timer()
    main()
    print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
