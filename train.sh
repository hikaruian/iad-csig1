CUDA_VISIBLE_DEVICES=0 python train.py \
    --train-root /home/kali/Downloads/project/iad-csig/data/Real-IAD/Train \
    --save-dir runs/smoke_test \
    --image-size 448 \
    --batch-size 1 \
    --grad-accum 1 \
    --epochs 1 \
    --num-workers 0




CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --train-root /home/kali/Downloads/project/iad-csig/data/Real-IAD/Train \
  --save-dir runs/inpformer_v3_t4x2 \
  --encoder dinov2reg_vit_base_14 \
  --encoder-source auto \
  --image-size 448 \
  --batch-size 2 \
  --grad-accum 4 \
  --epochs 200 \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --weight-decay 1e-6 \
  --inp-num 6 \
  --decoder-depth 8 \
  --gather-weight 0.2 \
  --soft-y 3.0 \
  --synthetic-prob 0.8 \
  --pixel-focal-weight 1.0 \
  --pixel-dice-weight 0.5 \
  --hard-negative-weight 0.03 \
  --boundary-weight 0.05 \
  --pixel-warmup-epochs 10 \
  --num-workers 2 \
  --amp


#OOM
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --train-root /home/kali/Downloads/project/iad-csig/data/Real-IAD/Train \
  --save-dir runs/inpformer_v3_t4x2 \
  --encoder dinov2reg_vit_base_14 \
  --image-size 448 \
  --batch-size 1 \
  --grad-accum 8 \
  --grad-checkpoint \
  --epochs 200 \
  --lr 1e-3 \
  --synthetic-prob 0.8 \
  --pixel-focal-weight 1.0 \
  --pixel-dice-weight 0.5 \
  --hard-negative-weight 0.03 \
  --boundary-weight 0.05 \
  --pixel-warmup-epochs 10 \
  --num-workers 2 \
  --amp



#smoke test
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --train-root /home/kali/Downloads/project/iad-csig/data/Real-IAD/Train \
  --save-dir runs/ddp_smoke \
  --encoder dinov2reg_vit_base_14 \
  --image-size 448 \
  --batch-size 1 \
  --grad-accum 1 \
  --epochs 1 \
  --num-workers 0 \
  --amp

