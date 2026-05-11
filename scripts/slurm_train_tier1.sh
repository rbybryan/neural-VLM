#!/bin/bash
#SBATCH --job-name=vlm_tier1
#SBATCH --partition=agcichy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=72:00:00
#SBATCH --qos=standard
#SBATCH --output=outputs/tier1/slurm_%j.log
#SBATCH --error=outputs/tier1/slurm_%j.err

set -euo pipefail

PROJECT_DIR="/home/byrong/projects/VLM-custom"
CONDA_ENV="vlm_custom"
TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"

DATA_JSON="${PROJECT_DIR}/data/train/llava_instruct_150k.json"
IMAGE_ROOT="/scratch/ccn_datasets/ms_coco/train2017"
OUTPUT_DIR="${PROJECT_DIR}/outputs/tier1"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

source /trinity/shared/easybuild/software/Anaconda3/2024.06-1/bin/activate
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1

# HF cache on scratch to avoid filling home quota
export HF_HOME="/scratch/ccn_datasets/.hf_cache_${USER}"
mkdir -p "${HF_HOME}"

echo "=== VLM tier-1 training ==="
echo "Job:   ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo ""

python scripts/train.py \
    --config configs/main.yaml \
    --data-json "${DATA_JSON}" \
    --image-root "${IMAGE_ROOT}" \
    --tokenizer "${TOKENIZER}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-steps 50000 \
    --batch-size 4 \
    --num-workers 4 \
    --learning-rate 1e-4 \
    --save-every 5000 \
    --log-every 50

echo ""
echo "Finished: $(date)"
