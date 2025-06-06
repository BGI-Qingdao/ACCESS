#!/bin/bash
path_structures="../pdb"
feature_file="../features.xml"
path_compute="./pdb_rosetta"
processed_files_log="../processed_files.log"
error_log="../miss_pdb.txt"  

cd $path_compute

while read -r pet_file; do
    
    original_file="${path_structures}/${pet_file}"
    base_name=$(echo "$pet_file" | sed -E 's/(.*)\.pdb(\.gz)?/\1_0001.pdb/')
    processed_file="${path_compute}/${base_name}"
    
    # Check if the processed file exists
    if [ ! -f "$processed_file" ]; then
        # If the file does not exist, run the script to process the original file

    	../rosetta.binary.linux.release-371/main/source/bin/rosetta_scripts.static.linuxgccrelease -s "$original_file" -parser:protocol "$feature_file" -overwrite

        if [ $? -ne 0 ]; then
            echo "$pet_file" >> "$error_log"  
            rm -f "$processed_file"           
            echo "error: $pet_file processing failed. Residual files have been cleared"

        fi
    else
        echo "$pet_file" >> "$processed_files_log"
        echo "File $processed_file already processed, skipping..."
    fi
done < $1
