cfg_name=cfg_train_gae_ms1m
config=gae/configs/$cfg_name.py

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# train
python gae/main.py \
    --config $config \
    --phase 'train'

# test
# load_from=data/work_dir/$cfg_name/latest.pth
# python gae/main.py \
#     --config $config \
#     --phase 'test' \
#     --load_from $load_from \
#     --save_output \
#     --force
