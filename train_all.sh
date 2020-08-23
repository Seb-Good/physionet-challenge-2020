#!/usr/bin/env bash
for FOLD in 0 1 2 3 4 5
do
    python main.py --start_fold $FOLD --n_epochs 100 --batch_size 64 --lr 0.001 --gpu 0
done


