import numpy as np
import pandas as pd
import json
import gzip
from pathlib import Path
import sys
import os

# oneway example (3 files: client side, server side, and network info):
# python3 make-parquet.py client/cl_104232_55500_2023725_15730.json.gz server/se_1721608_55500_2023725_15730.json.gz client/mni_20230725_150725.json.gz trip=uplink
#
# roundtrip example (2 files: client side and network info) note that "none" must be used:
# python3 make-parquet.py cl_104232_55500_2023725_15730.json.gz none mni_20230725_150725.json.gz
# 
# onway example no network info
# python3 make-parquet.py cl_104232_55500_2023725_15730.json.gz se_1721608_55500_2023725_15730.json.gz none trip=uplink device=adv01


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
            print(f"Ignoring invalid argument: {arg}")
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
        outputdir : str = "",
    ):

    cl_latencydata = opengzip(cl_latencyfile)
    
    # read gz files into dict
    if cl_netinfofile == "none":
        netinfodata = None
    else:
        netinfodata = opengzip(cl_netinfofile)
        print(f"Network info file contains {len(netinfodata)} records")

    # read gz files into dict
    if se_latencyfile == "none":
        se_latencydata = None
    else:
        se_latencydata = opengzip(se_latencyfile)

    # preprocess netinfo
    if netinfodata:
        for i in range(len(netinfodata)):
            if not netinfodata[i]['Band']:
                del netinfodata[i]

    if "round_trips" in cl_latencydata:
        cl_latency_samples = cl_latencydata["round_trips"]
        print(f"Client latency file contains {len(cl_latency_samples)} records")
    else:
        print("Error: client latency file does not contain round_trips.")
        return
    
    exclude_keys = ["oneway_trips","round_trips"]
    cl_latency_config = {k: cl_latencydata[k] for k in set(list(cl_latencydata.keys())) - set(exclude_keys)}
    se_latency_config = {}

    if se_latencydata:
        if "oneway_trips" in se_latencydata:
            print("Processing oneway latency measurements...")

            se_latency_samples = se_latencydata["oneway_trips"]
            print(f"Server latency file contains {len(se_latency_samples)} records")
    
            se_latency_config = {k: se_latencydata[k] for k in set(list(se_latencydata.keys())) - set(exclude_keys)}
        else:
            print("Error: server latency file does not contain oneway_trips.")
            return
    
    
    latency_config = {
        "server": se_latency_config, 
        "client": cl_latency_config
    }
    print(f"Measurement config:\n{json.dumps(latency_config, indent=4, sort_keys=True)}")

    # Match timestamps based on client measurements
    latency_timestamps = []
    for latency_sample in cl_latency_samples:
        if 'wall' in latency_sample['timestamps']['client']['send']:
            latency_timestamps.append(latency_sample['timestamps']['client']['send']['wall'])
    latency_timestamps = np.array(latency_timestamps)

    results = []
    if netinfodata:
        # match net info timestamps
        netinfo_timestamps = np.array([measurement['timestamp'] for measurement in netinfodata])
        netinfo_matched_idxs = np.array([np.argmin(abs(timestamp-netinfo_timestamps)) for timestamp in latency_timestamps])

        # append net info measurements
        for idx,midx in enumerate(netinfo_matched_idxs):
            results.append({**cl_latency_samples[idx], **{"netinfodata" : netinfodata[midx]}})
    else:
        results = cl_latency_samples

    # insert server timestamps if there is server latency data
    if se_latencydata:
        replace_list = ['timestamps', 'delay']
        for idx,latency_sample in enumerate(results):
            # find from the server latency samples with seqno
            seqno = latency_sample["seqno"]
            server_latency_sample = next((item for item in se_latency_samples if item['seqno'] == seqno), None)
            # update replace_list items if a record was found
            if server_latency_sample:
                # if delay is present in server records, means there are timestamps as well
                if server_latency_sample['delay']:
                    for item in replace_list:
                        latency_sample[item] = server_latency_sample[item]
            # update result in the list
            results[idx] = latency_sample
    
    for idx, result in enumerate(results):
        # Append packet size and multiple
        result['packet_length'] = latency_config['client']['config']['params']['length']
        result['packet_multiply'] = latency_config['client']['config']['params']['multiply']
        result['packet_interval'] = latency_config['client']['config']['params']['interval']

        # Append arbitrary arguments if there are any
        if arguments_dict:
            for key in arguments_dict:
                result[key] = arguments_dict[key]
        
        results[idx] = result

    # remove all branches in the dict
    results = pd.json_normalize(results, sep=".")

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
    print(f"Save results into {complete_parquetfile}")

    print(results.head(n=10).to_string(index=False))
    results.to_parquet(complete_parquetfile)

if __name__ == "__main__":

    args_num = len(sys.argv)
    if args_num < 4:
        print("no client latency file, server latency file, or network information file are provided.")
        exit(0)

    cl_latencyfile = str(sys.argv[1]) # Input client latency json.gz file
    print(f"client latency file: {cl_latencyfile}")

    se_latencyfile = str(sys.argv[2]) # Input server latency json.gz file
    print(f"server latency file: {se_latencyfile}")

    cl_netinfofile = str(sys.argv[3]) # Input network information json.gz file
    print(f"network information file: {cl_netinfofile}")

    if args_num > 4:
        # add arbitrary key=values to the results table
        command_line_args = sys.argv[4:]
        arguments_dict = parse_arguments_to_dict(command_line_args)
        print(f"arguments_dict: {arguments_dict}")
    else:
        print(f"no key=value arguments")
        arguments_dict = {}

    create_parquet_from_files(cl_latencyfile,se_latencyfile,cl_netinfofile,arguments_dict)
