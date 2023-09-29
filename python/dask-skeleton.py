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
from dask.distributed import Client
import pathlib
import argparse

# ==============================================================================
def main():
    # Get command line arguments
    cli = argparse.ArgumentParser()
    # Required Arguments
    cli.add_argument('-N', '--num-workers',    type=int,          required=True, help='The number of workers to use')
    cli.add_argument('-s', '--scheduler-file', type=pathlib.Path, required=True, help='The path to the scheduler file')
    # Optional Arguments
    # none yet, feel free to add your own
    args = cli.parse_args()

    # Setup the Dask cluster
    client = startup_dask(args.scheduler_file, args.num_workers)

    # Perform your computation
    # ...
    # ...  Output visualization
    # ...
    # End of Computation

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
