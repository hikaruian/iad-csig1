CUDA_VISIBLE_DEVICES=0 \
python infer.py \
  --test-root /kaggle/input/datasets/dickdickgo/real-iad/Test_A \
  --train-root /kaggle/input/datasets/dickdickgo/real-iad/Train \
  --ckpt runs/inpformer_v3/best.pth \
  --out-dir outputs/v3_raw \
  --zip outputs/v3_raw.zip \
  --samples-per-batch 1 \
  --max-ratio 0.001 \
  --sigma 0 \
  --no-legacy-refine \
  --no-view-gate \
  --amp


#TTA
CUDA_VISIBLE_DEVICES=0 \
python infer.py \
  --test-root /kaggle/input/datasets/dickdickgo/real-iad/Test_A \
  --ckpt runs/inpformer_v3/best.pth \
  --out-dir outputs/v3_tta \
  --zip outputs/v3_tta.zip \
  --samples-per-batch 1 \
  --max-ratio 0.001 \
  --sigma 0 \
  --tta-flip \
  --no-legacy-refine \
  --no-view-gate \
  --amp


#TTA
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
  --standalone \
  --nproc_per_node=2 \
  infer.py \
  --test-root /kaggle/input/datasets/dickdickgo/real-iad/Test_A \
  --ckpt runs/inpformer_v3/best.pth \
  --out-dir outputs/v3_tta \
  --zip outputs/v3_tta.zip \
  --samples-per-batch 1 \
  --max-ratio 0.001 \
  --sigma 0 \
  --tta-flip \
  --no-legacy-refine \
  --no-view-gate \
  --amp

