#!/bin/bash
#SBATCH --job-name=seal_rag_eval
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/slurm_%j.out

cd /gpfs/scratch/yg3030/AdaGATE/SEAL-RAG
uv run python evaluate.py
