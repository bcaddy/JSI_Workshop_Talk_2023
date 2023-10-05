#!/usr/bin/env bash

#SBATCH -A csc380
#SBATCH -J Orszag_tang_full_scale_analysis
#SBATCH -o /lustre/orion/ast181/proj-shared/rcaddy/JSI_Workshop_Talk_2023/data/otv_full_scale/%x-%j-analysis.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 32
#SBATCH --mail-user=r.caddy@pitt.edu
#SBATCH --mail-type=ALL

DASK_SCHEDULE_FILE=/lustre/orion/ast181/proj-shared/rcaddy/JSI_Workshop_Talk_2023/data/otv_full_scale/dask_schedule_file.json
DASK_NUM_WORKERS=$((SLURM_JOB_NUM_NODES*8))

export PYTHONPATH="${PYTHONPATH}:/lustre/orion/ast181/proj-shared/rcaddy/JSI_Workshop_Talk_2023/python"
export PATH="/ccs/home/rcaddy/miniconda_crusher/bin:$PATH"
source activate /ccs/proj/ast181/rcaddy/conda-envs/crusher/py-3.11

conda env list

# INTERFACE='--interface ib0' # For Andes
INTERFACE='' # For Crusher

srun --exclusive --ntasks=1 dask scheduler $INTERFACE --scheduler-file $DASK_SCHEDULE_FILE --no-dashboard --no-show &

#Wait for the dask-scheduler to start
sleep 30

srun --exclusive --ntasks=$DASK_NUM_WORKERS dask worker --scheduler-file $DASK_SCHEDULE_FILE --memory-limit='auto' --worker-class distributed.Worker $INTERFACE --no-dashboard --local-directory /ccs/home/rcaddy/ast181-orion/proj-shared/rcaddy/JSI_Workshop_Talk_2023/data/otv_full_scale/dask-scratch-space &

#Wait for workers to start
sleep 30

python -u ./dask-andes-runner.py --scheduler-file $DASK_SCHEDULE_FILE --num-workers $DASK_NUM_WORKERS --num-ranks=196 --gen-images=True --gen-video=True
#--cat-file=True --gen-images=True --gen-video=True

wait