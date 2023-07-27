import numpy as np
import pandas as pd
from loguru import logger
import json
import gzip
from pathlib import Path
import sys
import os

# oneway example (3 files: client side, server side, and network info):
# python3 makeparquet.py testfiles/client/cl_10-42-3-2_55500_20230726_171830.json.gz testfiles/server/se_172-16-0-8_55500_20230726_171830.json.gz testfiles/networkinfo/mni_20230726_171825.json.gz testfiles trip=uplink
#
# roundtrip example (2 files: client side and network info) note that "none" must be used:
# python3 makeparquet.py cl_104232_55500_2023725_15730.json.gz none mni_20230725_150725.json.gz testfiles 
# 
# onway example no network info
# python3 makeparquet.py cl_104232_55500_2023725_15730.json.gz se_1721608_55500_2023725_15730.json.gz none testfiles trip=uplink device=adv01


# Open gzip file storing a json
def opengzip(filename):
    with gzip.open(filename, 'r') as fin:            # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
    
    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

def parse_arguments_to_dict(args):
    argument_dict = {}
    for arg in args:
        # Split the argument by '=' to separate key and value
        key_value = arg.split('=')
        if len(key_value) == 2:
            key, value = key_value
            argument_dict[key] = value
        else:
            logger.warning(f"Ignoring invalid argument: {arg}")
    return argument_dict

def remove_first_element(input_string, separator='_'):
    # Split the string by the separator
    elements = input_string.split(separator)
    
    # Check if there are at least two elements in the list
    if len(elements) >= 2:
        # Remove the first element
        elements.pop(0)
        
        # Join the elements back into a string using the separator
        result_string = separator.join(elements)
        return result_string
    else:
        # Return the original string if there's only one element or an empty string
        return input_string

def create_parquet_from_files(
        cl_latencyfile : str,
        se_latencyfile : str,
        cl_netinfofile : str,
        arguments_dict : dict,
        outputdir : str = ".",
    ):

    
    # read netinfo file into dict
    if cl_netinfofile == "none":
        netinfodata = None
        logger.info(f"Network info file: none.")
    else:
        netinfodata = opengzip(cl_netinfofile)
        logger.info(f"Network info file contains {len(netinfodata)} records")

    # read server file into dict
    if se_latencyfile == "none":
        se_latencydata = None
        logger.info(f"Server latency file: none.")
    else:
        se_latencydata = opengzip(se_latencyfile)

    # read client file into dict
    try:
        cl_latencydata = opengzip(cl_latencyfile)
    except Exception as e:
        # Exception block to catch and handle any type of exception
        logger.error(f"Could not read client latency file: {e}")
        return
    
    if "round_trips" in cl_latencydata:
        cl_latency_samples = cl_latencydata["round_trips"]
        logger.info(f"Client latency file contains {len(cl_latency_samples)} records")
    else:
        logger.error("Error: client latency file does not contain round_trips.")
        return
    

    # extracting the configuration
    exclude_keys = ["oneway_trips","round_trips"]
    cl_latency_config = {k: cl_latencydata[k] for k in set(list(cl_latencydata.keys())) - set(exclude_keys)}
    se_latency_config = {}

    # is it oneway or roundtrip
    if se_latencydata:
        if "oneway_trips" in se_latencydata:
            logger.info("Processing oneway latency measurements...")

            se_latency_samples = se_latencydata["oneway_trips"]
            logger.info(f"Server latency file contains {len(se_latency_samples)} records")

            se_latency_config = {k: se_latencydata[k] for k in set(list(se_latencydata.keys())) - set(exclude_keys)}
        else:
            logger.error("Error: server latency file does not contain oneway_trips.")
            return
    
    
    latency_config = {
        "server": se_latency_config, 
        "client": cl_latency_config
    }
    logger.info(f"Measurement config:\n{json.dumps(latency_config, indent=4, sort_keys=True)}")

    # match timestamps based on client measurements
    latency_timestamps = []
    for latency_sample in cl_latency_samples:
        if 'wall' in latency_sample['timestamps']['client']['send']:
            latency_timestamps.append(latency_sample['timestamps']['client']['send']['wall'])
    latency_timestamps = np.array(latency_timestamps)

    results = []
    if netinfodata:
        logger.info(f"match net info timestamps with client timestamps")
        # match net info timestamps
        netinfo_timestamps = np.array([measurement['timestamp'] for measurement in netinfodata])
        netinfo_matched_idxs = np.array([np.argmin(abs(timestamp-netinfo_timestamps)) for timestamp in latency_timestamps])

        # append net info measurements
        for idx,midx in enumerate(netinfo_matched_idxs):
            results.append({**cl_latency_samples[idx], **{"netinfodata" : netinfodata[midx]}})
    else:
        results = cl_latency_samples
    
    # make a columnar dataframe for results
    logger.info(f"create results dataframe")
    results_df = pd.json_normalize(results, sep=".")
    results_df.set_index("seqno", inplace=True)
    del results

    # insert server timestamps if there is server latency data
    if se_latencydata:
        
        # make a columnar dataframe for server latency measurements
        logger.info(f"create server records dataframe")
        se_latency_df = pd.json_normalize(se_latency_samples, sep=".")
        # Set the "seqno" column as the index for fast access
        se_latency_df.set_index("seqno", inplace=True)
        del se_latency_samples
    
        # Update only certain columns in "results_df" from "se_latency_df"
        column_names_to_update = ['timestamps']
        column_names_to_add = ['delay.send']
        columns_to_update = []
        for name in column_names_to_update:
            res_column_names = [col for col in results_df.columns if col.startswith(name)]
            columns_to_update = [*columns_to_update, *res_column_names]
            
            se_column_names = [col for col in se_latency_df.columns if col.startswith(name)]
            columns_to_update = [*columns_to_update, *se_column_names]
        for name in column_names_to_add:
            columns_to_update = [*columns_to_update, *column_names_to_add]
        # remove redundant strings
        columns_to_update = list(set(columns_to_update))

        # Check and create columns with null values if they don't exist
        for column_name in columns_to_update:
            if column_name not in results_df.columns:
                results_df[column_name] = None

        logger.info(f"combine server records with client records on these columns {columns_to_update}")
        results_df.update(se_latency_df[columns_to_update])

    logger.info("add packet_length, packet_multiply, and packet_interval to results")
    results_df['packet_length'] = latency_config['client']['config']['params']['length']
    results_df['packet_multiply'] = latency_config['client']['config']['params']['multiply']
    results_df['packet_interval'] = latency_config['client']['config']['params']['interval']


    logger.info(f"add {arguments_dict} to results")
    # Append arbitrary arguments if there are any
    if arguments_dict:
        for key in arguments_dict:
            results_df[key] = arguments_dict[key]
    
    # make base name for the output file
    base_name = os.path.basename(cl_latencyfile)
    base_name, extensions = os.path.splitext(base_name)
    while extensions:
        base_name, extensions = os.path.splitext(base_name)
    base_name = remove_first_element(base_name)
    parquetfile = base_name + '.parquet'

    # Check if the outputdir exists, if not, create it (including any necessary parent directories)
    os.makedirs(outputdir, exist_ok=True)

    # Combine the outputdir and filename to get the complete address
    complete_parquetfile = os.path.join(outputdir, parquetfile)
    logger.info(f"Save results into {complete_parquetfile}")

    print(results_df.head(n=10).to_string(index=False))
    results_df.to_parquet(complete_parquetfile)

if __name__ == "__main__":

    args_num = len(sys.argv)
    if args_num < 4:
        logger.error("no client latency file, server latency file, or network information file are provided.")
        exit(0)

    cl_latencyfile = str(sys.argv[1]) # Input client latency json.gz file
    logger.info(f"client latency file: {cl_latencyfile}")

    se_latencyfile = str(sys.argv[2]) # Input server latency json.gz file
    logger.info(f"server latency file: {se_latencyfile}")

    cl_netinfofile = str(sys.argv[3]) # Input network information json.gz file
    logger.info(f"network information file: {cl_netinfofile}")

    outputdir = str(sys.argv[4])
    logger.info(f"output dir: {outputdir}")

    if args_num > 5:
        # add arbitrary key=values to the results table
        command_line_args = sys.argv[5:]
        arguments_dict = parse_arguments_to_dict(command_line_args)
        logger.info(f"arguments_dict: {arguments_dict}")
    else:
        logger.warning(f"no key=value arguments")
        arguments_dict = {}

    create_parquet_from_files(cl_latencyfile, se_latencyfile, cl_netinfofile, arguments_dict, outputdir)
