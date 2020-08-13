#!/usr/bin/env bash
for FOLD in 0 1 2 3 4 5
do
    python main.py --start_fold $FOLD --lr 0.001 --batch_size 2 --n_epochs 1
done
