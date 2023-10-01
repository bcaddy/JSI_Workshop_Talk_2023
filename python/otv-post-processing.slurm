#!/usr/bin/env bash

#SBATCH -A csc380
#SBATCH -J Orszag_tang_small_scale_analysis
#SBATCH -o /lustre/orion/ast181/scratch/rcaddy/JSI_Workshop_Talk_2023/data/otv_small_scale/%x-%j-analysis.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 30
#SBATCH --mail-user=r.caddy@pitt.edu
#SBATCH --mail-type=ALL

DASK_SCHEDULE_FILE=/lustre/orion/ast181/scratch/rcaddy/JSI_Workshop_Talk_2023/data/otv_small_scale/dask_schedule_file.json
DASK_NUM_WORKERS=120

srun --exclusive --ntasks=1 dask scheduler --interface ib0 --scheduler-file $DASK_SCHEDULE_FILE --no-dashboard --no-show &

#Wait for the dask-scheduler to start
sleep 30

srun --exclusive --ntasks=$DASK_NUM_WORKERS dask worker --scheduler-file $DASK_SCHEDULE_FILE --memory-limit='auto' --worker-class distributed.Worker --interface ib0 --no-dashboard --local-directory /ccs/home/rcaddy/ast181-orion/scratch/rcaddy/JSI_Workshop_Talk_2023/data/otv_small_scale/dask-scratch-space &

#Wait for workers to start
sleep 30

python -u ./dask-andes-runner.py --scheduler-file $DASK_SCHEDULE_FILE --num-workers $DASK_NUM_WORKERS --cat-file=True --gen-images=True --gen-video=True

wait