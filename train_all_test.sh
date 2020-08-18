#!/usr/bin/env bash
for FOLD in 0 1 2 3 4 5
do
    python main.py --start_fold $FOLD --n_epochs 1 --batch_size 128 --lr 0.001 --gpu 3,4,5 --downsample True
done
