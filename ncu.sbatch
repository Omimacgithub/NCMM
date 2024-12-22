#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run CUDA applications
# on CESGA's FT-III system.
#----------------------------------------------------
#SBATCH -J gpu_job       # Job name
#SBATCH -o gpu_job.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e gpu_job.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32            # Cores per task requested (1 task job)
# Needed 32 cores per A100 demanded
#SBATCH --mem-per-cpu=3G # memory per core demanded
#SBATCH --gres=gpu:a100  # Options for requesting 1GPU
#SBATCH -t 00:30:00      # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --exclusive

mkdir outputs

# Run the CUDA application
module load cuda/12.2.0

CUDA_BLK=(32 64 128 256 512 1024)

make MM
for cb in "${CUDA_BLK[@]}"; do
  echo "-------------------"
  echo ""
  echo "-------------------"
  echo "-- CUDA_BLK = $cb --"
  echo "-------------------"
  echo ""
  echo "-------------------"
  ncu --config-file on -o outputs/MM$cb MM 512 $cb
done

make MMo
for cb in "${CUDA_BLK[@]}"; do
  echo "-------------------"
  echo ""
  echo "-------------------"
  echo "-- CUDA_BLK = $cb --"
  echo "-------------------"
  echo ""
  echo "-------------------"
  ncu --config-file on -o outputs/MMo$cb MMo 512 $cb
done

make MMo1
for cb in "${CUDA_BLK[@]}"; do
  echo "-------------------"
  echo ""
  echo "-------------------"
  echo "-- CUDA_BLK = $cb --"
  echo "-------------------"
  echo ""
  echo "-------------------"
  ncu --config-file on -o outputs/MMo1$cb MMo1 512 $cb
done

make MMo2
for cb in "${CUDA_BLK[@]}"; do
  echo "-------------------"
  echo ""
  echo "-------------------"
  echo "-- CUDA_BLK = $cb --"
  echo "-------------------"
  echo ""
  echo "-------------------"
  ncu --config-file on -o outputs/MMo2$cb MMo2 512 $cb
done

