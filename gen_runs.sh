#!/bin/bash

# Define the arrays for p and temp values
p_values=(0.2 0.5 0.8)
temp_values=(1 2 5)

# Loop over temp values
for temp in "${temp_values[@]}"; do
    # Loop over p values
    for p in "${p_values[@]}"; do
        echo "Running with temp=$temp and p=$p"
        python -m cs336_basics.generate \
            --checkpoint-path="ckpt_final/00099.pt" \
            --prompt="Once upon a time, there was a little boy named" \
            --temp="$temp" \
            --p="$p" \
            --device=mps
    done
done