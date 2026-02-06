#!/bin/bash

#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=OpenVLA-OFT-CTRL-PLUS
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --requeue

# Ctrl-World Plus Training Script (Bridge-style SimpleVLA-RL config)
# This script trains on combined iter_sft_bridge + bridge_dataset
#
# Key configuration:
# - Use LLaMA-2 LM head for actions (--use_l1_regression False)
# - Single camera input (--num_images_in_input 1)
# - No proprioceptive state (--use_proprio False)
# - Action chunking/parallel decoding follow Bridge constants (dataset name contains "bridge")

# ============= CONFIGURATION =============

# Number of GPUs to use
NUM_GPUS=1

# Paths
DATA_ROOT_DIR="/projects/work/yang-lab/projects/pretrain_world_model/oxe_tfds_raw"
RUN_ROOT_DIR="/scratch/ys4907/checkpoints/openvla-oft/iter_sft_plus_sft_new_lora"

# Dataset
DATASET_NAME="iter_sft_plus"

# Training hyperparameters
BATCH_SIZE=32           # Per-GPU batch size
LEARNING_RATE=5e-4
MAX_STEPS=200000
NUM_STEPS_BEFORE_DECAY=100000
SAVE_FREQ=10000

# LoRA
LORA_RANK=32

# Weights & Biases (optional)
WANDB_ENTITY="yixiangsun"
WANDB_PROJECT="openvla-ctrl-world-plus-sft"
RUN_ID_NOTE="iter_sft_plus--lm_head--single_cam--no_proprio"

# ============= LAUNCH TRAINING =============

echo "============================================"
echo "Starting OpenVLA-OFT Training on Ctrl-World Plus"
echo "============================================"
echo "Configuration:"
echo "  Dataset: ${DATASET_NAME}"
echo "  Data root: ${DATA_ROOT_DIR}"
echo "  Checkpoint dir: ${RUN_ROOT_DIR}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Max steps: ${MAX_STEPS}"
echo "  LoRA rank: ${LORA_RANK}"
echo "============================================"
echo ""

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} vla-scripts/finetune.py \
  --vla_path /projects/work/yang-lab/projects/wmrl/openvla-oft-bridge-sft-20000-ctrl-world-sft \
  --data_root_dir ${DATA_ROOT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --num_steps_before_decay ${NUM_STEPS_BEFORE_DECAY} \
  --max_steps ${MAX_STEPS} \
  --save_freq ${SAVE_FREQ} \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank ${LORA_RANK} \
  --wandb_entity ${WANDB_ENTITY} \
  --wandb_project ${WANDB_PROJECT} \
  --run_id_note ${RUN_ID_NOTE} \
  --resume False

echo ""
echo "============================================"
echo "Training completed!"
echo "Checkpoints saved to: ${RUN_ROOT_DIR}"
echo "============================================"
