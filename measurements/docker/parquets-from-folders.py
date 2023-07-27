#!/usr/bin/env python3

import os
import sys
import subprocess
from loguru import logger
from datetime import datetime, timedelta
from makeparquet import create_parquet_from_files, parse_arguments_to_dict

# Function to check if two timestamps are within 1 minute of each other
def is_within_1_minute(t1, t2):
    diff = abs((t2 - t1).total_seconds())
    return diff <= 60


def extract_timestamp_from_filename(filename):
    # Remove the file extension
    base_name, extensions = os.path.splitext(filename)
    while extensions:
        base_name, extensions = os.path.splitext(base_name)
    filename_without_extension = base_name

    # Extract the timestamp parts from the filename
    timestamp_parts = filename_without_extension.split('_')[-2:]

    # Combine the timestamp parts into a datetime object
    try:
        timestamp = datetime.strptime(' '.join(timestamp_parts), '%Y%m%d %H%M%S')
        return timestamp
    except ValueError:
        # If the timestamp cannot be parsed correctly, return None or handle the error as per your requirement
        return None

def main():

    # Check if the correct number of arguments is provided
    args_num = len(sys.argv)
    if args_num < 5:
        logger.error("Usage: {} <output dir> <folder1> <folder2> <folder3>".format(sys.argv[0]))
        sys.exit(1)

    # Get the folder paths from the command-line arguments
    outputdir = sys.argv[1]
    folder1 = sys.argv[2]
    folder2 = sys.argv[3]
    folder3 = sys.argv[4]

    if args_num > 5:
        # add arbitrary key=values to the results table
        command_line_args = sys.argv[5:]
        arguments_dict = parse_arguments_to_dict(command_line_args)
        logger.info(f"arguments_dict: {arguments_dict}")
    else:
        logger.warning(f"no key=value arguments")
        arguments_dict = {}

    # Get the list of files from both folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2) if folder2 != "none" else []
    files3 = os.listdir(folder3) if folder3 != "none" else []

    # Loop through each file in folder1
    for file1 in files1:
        # Get the full path and creation time of the file in folder1
        full_path1 = os.path.join(folder1, file1)
        # creation_time1 = os.path.getctime(full_path1)
        creation_time1 = extract_timestamp_from_filename(file1)

        #print(f"Processing file in {folder1}: {full_path1} (Created at: {datetime.fromtimestamp(creation_time1)})")
        logger.info(f"Processing file in {folder1}: {full_path1} (Created at: {creation_time1})")

        # check server latency folder (folder2)

        # Variable to check if a match is found
        match_found_f2 = False
        match_f2 = 'none'

        # Loop through each file in folder2
        for file2 in files2:
            # Get the full path and creation time of the file in folder2
            full_path2 = os.path.join(folder2, file2)
            #creation_time2 = os.path.getctime(full_path2)
            creation_time2 = extract_timestamp_from_filename(file2)

            # Check if the creation times are within 1 minute of each other
            if is_within_1_minute(creation_time1, creation_time2):
                logger.info("\tMatching pair:")
                #print(f"\tFile in {folder2}: {full_path2} (Created at: {datetime.fromtimestamp(creation_time2)})\n")
                logger.info(f"\tFile in {folder2}: {full_path2} (Created at: {creation_time2})\n")

                # Mark match_found_f2 as True to indicate a match is found
                match_found_f2 = True
                match_f2 = full_path2

                # Remove the matched file from the files2 list to avoid reprocessing
                files2.remove(file2)
                break

        # If no match is found in folder2 for a file in folder1, print a warning
        if not match_found_f2:
            logger.warning(f"\tNo match found for file in {folder1}, in {folder2}: {full_path1}")

        # check network info folder (folder3)

        # Variable to check if a match is found
        match_found_f3 = False
        match_f3 = 'none'

        # Loop through each file in folder3
        for file3 in files3:
            # Get the full path and creation time of the file in folder3
            full_path3 = os.path.join(folder3, file3)
            #creation_time3 = os.path.getctime(full_path3)
            creation_time3 = extract_timestamp_from_filename(file3)


            # Check if the creation times are within 1 minute of each other
            if is_within_1_minute(creation_time1, creation_time3):
                logger.info("\tMatching pair:")
                #print(f"\tFile in {folder3}: {full_path3} (Created at: {datetime.fromtimestamp(creation_time3)})\n")
                logger.info(f"\tFile in {folder3}: {full_path3} (Created at: {creation_time3})\n")

                # Mark match_found_f3 as True to indicate a match is found
                match_found_f3 = True
                match_f3 = full_path3

                # Remove the matched file from the files3 list to avoid reprocessing
                files3.remove(file3)
                break

        # If no match is found in folder3 for a file in folder1, print a warning
        if not match_found_f3:
            logger.warning(f"\tNo match found for file in {folder1}, in {folder3}: {full_path1}")

        # run the python command
        create_parquet_from_files(full_path1, match_f2, match_f3, arguments_dict, outputdir)
        #python_script = ["python3", "make-parquet.py", full_path1, match_f2, match_f3] + sys.argv[4:]
        #subprocess.run(python_script)

if __name__ == "__main__":
    main()
