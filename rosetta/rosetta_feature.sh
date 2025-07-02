path_structures="../pdb"
feature_file="../features.xml"
path_compute="./pdb_rosetta"

input_file_list="path/pdb_list/pdb_list.txt"

cd $path_compute

while read -r pet_file; do
    
    original_file="${path_structures}/${pet_file}"
    base_name=$(echo "$pet_file" | sed -E 's/(.*)\.pdb(\.gz)?/\1_0001.pdb/')
    processed_file="${path_compute}/${base_name}"
    
    # Check if the processed file exists
    if [ ! -f "$processed_file" ]; then
        # If the file does not exist, run the script to process the original file
        ../rosetta.binary.linux.release-371/main/source/bin/rosetta_scripts.static.linuxgccrelease -s "$original_file" -parser:protocol "$feature_file" -overwrite
    else
        echo "File $processed_file already processed, skipping..."
    fi
done < "$input_file_list"
