#!/bin/bash

for dir in `ls ./data/raw_data/`; do
    echo "python3 ./bin/2_resave.py --ep_dir ./data/raw_data/${dir}"
    python3 ./bin/2_resave.py --ep_dir ./data/raw_data/${dir}
done
