#!/bin/bash
#SBATCH --job-name=vlm_tier1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# Submit from the repository root with the environment already activated:
# IMAGE_ROOT=/path/to/coco/train2017 DATA_JSON=/path/to/train.json \
#   sbatch --partition=YOUR_PARTITION scripts/slurm_train_tier1.sh
# Pass any site-specific GPU type, account, QoS, or time limit to sbatch.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-1.5B-Instruct}"

DATA_JSON="${DATA_JSON:?Set DATA_JSON to the training manifest}"
IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to the training image directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/tier1}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1

# Override HF_HOME for scratch storage if needed.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
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
