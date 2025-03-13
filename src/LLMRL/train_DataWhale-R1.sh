export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch \
    --num_processes 2 \
    --config_file deepspeed_zero3.yaml \
    train_Datawhale-R1.py \
    --config Datawhale-R1.yaml

