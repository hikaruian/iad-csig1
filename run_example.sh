#!/usr/bin/env bash
# 2×T4 16GB, single node. Edit the two paths first.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}


# global batch = 2 GPU × 4 × accum 2 = 16 (paper)
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --train-root "./data/Real-IAD/Train" \
  --save-dir runs/inpformer_b14 \
  --encoder dinov2reg_vit_base_14 \
  --image-size 448 \
  --inp-num 6 \
  --epochs 200 \
  --batch-size 4 \
  --grad-accum 2 \
  --num-workers 2 \
  --amp

torchrun --standalone --nnodes=1 --nproc_per_node=2 infer.py \
  --test-root "./data/Real-IAD/Test_A" \
  --ckpt runs/inpformer_b14/best.pth \
  --train-root "./data/Real-IAD/Train" \
  --out-dir submission \
  --zip my_submission.zip \
  --samples-per-batch 2 \
  --sigma 2.5 \
  --gamma 1.4 \
  --refine \
  --fg-gate \
  --max-ratio 0.01 \
  --reduce max \
  --tta-flip \
  --amp
