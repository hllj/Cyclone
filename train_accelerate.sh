export WANDB_PROJECT=paligemma
export CUTLASS_PATH=/home/vanhop/Cyclone/cutlass

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py configs/paligemma/paligemma_v1.yaml

