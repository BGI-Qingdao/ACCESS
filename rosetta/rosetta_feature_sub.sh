#!/bin/bash

path_compute="./pdb_list"
total_workers=64  # Node CPU cores

# Parallel execution of all tasks (with 1 CPU bound to each task)
for i in $(seq -w 0 $((total_workers-1)) ); do
    current_num=$((10#${i} + 0))
    formatted_i=$(printf "%03d" "${current_num}")
    list_file="$path_compute/list_files_part_${formatted_i}.txt"
    
    if [ -f "$list_file" ]; then
        # Binding CPU with taskset
        taskset -c $((10#${i})) ./rosetta_feature_parallel.sh "$list_file" &
    else
        echo "ERROR: $list_file missing" >&2
    fi
done

wait
echo "All workers finished"

