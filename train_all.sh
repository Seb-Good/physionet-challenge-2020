#!/usr/bin/env bash
for FOLD in 0 2 3 4 5
do
    python main.py --start_fold $FOLD --lr 0.001 --batch_size 128 --n_epochs 100
done
