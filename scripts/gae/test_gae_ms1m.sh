config=gae/configs/cfg_test_gae_ms1m.py
load_from=data/pretrained_models/pretrained_gae_ms1m.pth

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python gae/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --force
