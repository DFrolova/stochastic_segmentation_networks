#!/bin/bash

python ssn/train.py --job-dir /shared/experiments/ood_playground/ssn/rank_10_mc_20_patch_110_cc359/train \
        --config-file assets/config_files/rank_10_mc_20_patch_110_cc359.json \
        --train-csv-path assets/cc359_data/data_index_train.csv \
        --valid-csv-path assets/cc359_data/data_index_valid.csv \
        --num-epochs 100 \
        --device 0 \
        --random-seeds "1"
