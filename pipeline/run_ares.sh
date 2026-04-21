#!/bin/bash
#SBATCH --job-name=ares_eval
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/slurm_ares_%j.out

source ~/.bashrc
conda activate ares-env
cd /gpfs/scratch/yg3030/AdaGATE/SEAL-RAG
python -u run_ares.py --input results_seal_rag_k3_L1_n20_20260324_194727.json
