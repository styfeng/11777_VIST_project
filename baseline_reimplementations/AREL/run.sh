#!/usr/bin/env bash

source /projects/tir5/users/cnariset/miniconda3/bin/activate
source activate py27

# python train.py --id 777_XE --data_dir ../DATADIR --start_rl -1
# python train_AREL.py --id 777_AREL --start_from_model data/save/777_XE/model-best.pth --data_dir ../DATADIR \
#     --decoding_method_DISC greedy --workers 8 --learning_rate 0.0002 --G_iter 50 --D_iter 50 \
#     --save_checkpoint_every 100 --beam_size 3

python train.py --id 777_XE --data_dir ../DATADIR --start_from_model data/save/777_XE/model-best.pth --option test
python train_AREL.py --id 777_AREL --start_from_model data/save/777_AREL/model-best.pth --data_dir ../DATADIR \
    --decoding_method_DISC greedy --workers 8 --beam_size 3 --option test
