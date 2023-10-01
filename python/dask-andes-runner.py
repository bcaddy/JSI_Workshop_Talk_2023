#!/usr/bin/env python3
"""
This is the skeleton for how to run a Dask script on Andes at the OLCF. The CLI
commands required are in the docstring at the top, major Dask steps are in
functions, and `main` is mostly empty with a clear area on where to do your
computations.

Requirements:
- Verified working with Dask v2023.6.0
- Install graphviz for python
  - 'conda install -c conda-forge python-graphviz graphviz'
- Make sure your version of msgpack-python is at least v1.0.5; v1.0.3 had a bug
  - `conda install -c conda-forge msgpack-python=1.0.5`

Notes:
- This is entirely focused on getting Dask to run on Andes. Other systems will
  likely need similar steps but not identical
- Between each python script the Dask scheduler and workers need to be restarted.
- "--interface ib0" does not seem to be required but likely does improve transfer speeds
- It likes to spit out lots of ugly messages on shutdown that look like something
  failed. Odds are that it worked fine and just didn't shutdown gracefully

################################################################################
#!/usr/bin/env bash

#SBATCH -A <allocation here>
#SBATCH -J <job name>
#SBATCH -o <slurm output file>/%x-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 30
#SBATCH --mail-user=<your email>
#SBATCH --mail-type=ALL

# Setup some parameters
DASK_SCHEDULE_FILE=$(pwd)/dask_schedule_file.json
DASK_NUM_WORKERS=4

srun --exclusive --ntasks=1 dask scheduler --interface ib0 --scheduler-file $DASK_SCHEDULE_FILE --no-dashboard --no-show &

#Wait for the dask-scheduler to start
sleep 10

srun --exclusive --ntasks=$DASK_NUM_WORKERS dask worker --scheduler-file $DASK_SCHEDULE_FILE --memory-limit='auto' --worker-class distributed.Worker --interface ib0 --no-dashboard --local-directory $(pwd)/dask-scratch-space &

#Wait for workers to start
sleep 10

python -u ./olcf-docs.py --scheduler-file $DASK_SCHEDULE_FILE --num-workers $DASK_NUM_WORKERS

wait
################################################################################
"""

import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
from dask import graph_manipulation
import pathlib
import argparse
import numpy as np

import cat_slice
import heatmap

# ==============================================================================
def main():
    # Get command line arguments
    cli = argparse.ArgumentParser()
    # Required Arguments
    cli.add_argument('-N', '--num-workers',    type=int,          required=True, help='The number of workers to use')
    cli.add_argument('-s', '--scheduler-file', type=pathlib.Path, required=True, help='The path to the scheduler file')
    cli.add_argument('-r', '--num-ranks',      type=int,          required=True, help='The number of ranks cholla was run with')
    # Optional Arguments
    cli.add_argument('--cat-files',  type=bool, default=False, help='Concatenate the data files.')
    cli.add_argument('--gen-images', type=bool, default=False, help='Generate the images.')
    cli.add_argument('--gen-video',  type=bool, default=False, help='Convert the images to videos.')
    # none yet, feel free to add your own
    args = cli.parse_args()

    # Setup the Dask cluster
    client = startup_dask(args.scheduler_file, args.num_workers)

    # Work to do
    outputs_to_work_on = np.arange(0,715)
    num_ranks = args.num_ranks

    root_directory        = pathlib.Path('/lustre/orion/ast181/scratch/rcaddy/JSI_Workshop_Talk_2023/data/otv_small_scale')
    source_directory      = root_directory / 'uncat_data'
    concat_file_directory = root_directory / 'concat_data'
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
                                                                output_number=output,
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
        video_task = dask.delayed(heatmap.make_video)(png_file_directory, video_file_directory, field, fps=24)

        if args.cat_files or args.gen_images:
            video_task = dask.graph_manipulation.bind(video_task, work_to_do[:pre_video_idx])
        if args.gen_video:
            work_to_do.append(video_task)

    # Save the task graph
    dask.visualize(*work_to_do, filename=str(root_directory/'dask-task-graph.pdf'))

    # Execute the work
    dask.compute(*work_to_do)

    # Shutdown the Dask cluster
    shutdown_dask(client)
# ==============================================================================

# ==============================================================================
def startup_dask(scheduler_file, num_workers):
    # Connect to the dask-cluster
    client = Client(scheduler_file=scheduler_file)
    print('client information ', client)

    # Block until num_workers are ready
    print(f'Waiting for {num_workers} workers...')
    client.wait_for_workers(n_workers=num_workers)

    num_connected_workers = len(client.scheduler_info()['workers'])
    print(f'{num_connected_workers} workers connected')

    return client
# ==============================================================================

# ==============================================================================
def shutdown_dask(client):
    print('Shutting down the cluster')
    workers_list = list(client.scheduler_info()['workers'])
    client.retire_workers(workers_list, close_workers=True)
    client.shutdown()
# ==============================================================================

if __name__ == '__main__':
    main()
