#!/usr/bin/env bash

#SBATCH -A csc380
#SBATCH -J Orszag_tang_full_scale
#SBATCH -o /ccs/home/rcaddy/ast181-orion/scratch/rcaddy/JSI_Workshop_Talk_2023/data/otv_full_scale/%x-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 25
#SBATCH --mail-user=r.caddy@pitt.edu
#SBATCH --mail-type=ALL

job_root=/ccs/home/rcaddy/ast181-orion/scratch/rcaddy/JSI_Workshop_Talk_2023
cd $job_root

source ./cholla/builds/setup.frontier.cce.sh

# For restarting if needed
# RESTART_ARGS=''
RESTART_ARGS="init=Read_Grid nfile=500 indir=${job_root}/data/otv_full_scale/uncat_data/"

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=196 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest ${job_root}/cholla/bin/cholla.mhd.frontier ${job_root}/orszag_tang_vortex.txt outdir="${job_root}/data/otv_full_scale/uncat_data/" $RESTART_ARGS
