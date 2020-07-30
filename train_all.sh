#!/usr/bin/env bash
for FOLD in 0 1 2 3 4
do
    python main.py --fold $FOLD
done
