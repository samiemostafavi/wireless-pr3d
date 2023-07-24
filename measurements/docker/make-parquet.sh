#!/bin/bash

# if there is file in the first folder that there is no match for it in the second folder, this script will throw a warning
# folder1: latency samples
# folder2: network info

# Function to check if two timestamps are within 1 minute of each other
function is_within_1_minute() {
    local t1=$1
    local t2=$2

    local diff=$((t2 - t1))
    local abs_diff=${diff#-}

    if [ $abs_diff -le 60 ]; then
        return 0  # Within 1 minute
    else
        return 1  # Not within 1 minute
    fi
}

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <folder1> <folder2>"
    exit 1
fi

# Get the folder paths from the command-line arguments
folder1="$1"
folder2="$2"

# Get the list of files from both folders
files1=$(ls -1 "$folder1")
files2=$(ls -1 "$folder2")

# skip the first 2 arguments
shift 2;

# Loop through each file in folder1
for file1 in $files1; do
    # Get the full path and creation time of the file in folder1
    full_path1=$(realpath "$folder1/$file1")
    creation_time1=$(stat -c %Y "$full_path1")

    # Variable to check if a match is found
    match_found=0

    # Loop through each file in folder2
    for file2 in $files2; do
        # Get the full path and creation time of the file in folder2
        full_path2=$(realpath "$folder2/$file2")
        creation_time2=$(stat -c %Y "$full_path2")

        # Check if the creation times are within 1 minute of each other
        if is_within_1_minute $creation_time1 $creation_time2; then
            echo "Matching pair:"
            echo "File in $folder1: $full_path1 (Created at: $(date -d @$creation_time1))"
            echo "File in $folder2: $full_path2 (Created at: $(date -d @$creation_time2))"
            echo
	
	    python3 make-parquet.py "$full_path1" "$full_path2" "$@"

            # Mark match_found as 1 to indicate a match is found
            match_found=1

            # Remove the matched file from the files2 list to avoid reprocessing
            files2=$(echo "$files2" | grep -v "$file2")
            break
        fi
    done

    # If no match is found for a file in folder1, print a warning
    if [ $match_found -eq 0 ]; then
        echo "Warning: No match found for file in $folder1: $full_path1"
    fi
done

