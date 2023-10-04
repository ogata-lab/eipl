#!/bin/bash

for dir in `ls ./data/raw_data/`; do
    echo "python3 ./bin/3_check_playback_data.py ./data/raw_data/${dir}/state_resave.npz"
    python3 ./bin/3_check_playback_data.py ./data/raw_data/${dir}/state_resave.npz
done
