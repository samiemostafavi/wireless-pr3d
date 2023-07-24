import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import gzip
from pathlib import Path
import sys
import os

# example
# python3 make-parquet.py 172-16-0-8-36970_7-18-23-26-32.json.gz adv01ul_20230718_232628.json.gz trip=uplink device=adv01

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

latencyfile = str(sys.argv[1]) # Input latency json.gz file
netinfofile = str(sys.argv[2]) # Input network information json.gz file

# add arbitrary key=values to the results table
command_line_args = sys.argv[3:]
arguments_dict = parse_arguments_to_dict(command_line_args)
print(f"arguments_dict: {arguments_dict}")

# read gz files into dict
latencydata = opengzip(latencyfile)
netinfodata = opengzip(netinfofile)
print(f"Network info file contains {len(netinfodata)} records")

# preprocess netinfo
for i in range(len(netinfodata)):
    if not netinfodata[i]['Band']:
        del netinfodata[i]

if "oneway_trips" in latencydata:
    print("Processing: oneway trips")

    latency_samples = latencydata["oneway_trips"]
    print(f"Latency file contains {len(latency_samples)} records")

    
elif "round_trips" in latencydata:
    print("Processing: round trips")
    
    latency_samples = latencydata["round_trips"]

else:
    print("corrupt latency json")
    exit(0)

exclude_keys = ["oneway_trips","round_trips"]
latency_config = {k: latencydata[k] for k in set(list(latencydata.keys())) - set(exclude_keys)}
print(f"Latency config: {latency_config}")

# Match timestamps
latency_timestamps = []
for latency_sample in latency_samples:
    if 'wall' in latency_sample['timestamps']['client']['send']:
        latency_timestamps.append(latency_sample['timestamps']['client']['send']['wall'])

latency_timestamps = np.array(latency_timestamps)

#latency_timestamps = np.array([latency_sample['timestamps']['client']['send']['wall'] for latency_sample in latency_samples]) # list of timestamps
netinfo_timestamps = np.array([measurement['timestamp'] for measurement in netinfodata])
time_matched_idxs = np.array([np.argmin(abs(timestamp-netinfo_timestamps)) for timestamp in latency_timestamps])

results = []
for idx,midx in enumerate(time_matched_idxs):
    results.append({**latency_samples[idx], **netinfodata[midx]})

results = pd.json_normalize(results, sep=".")

base_name, extensions = os.path.splitext(latencyfile)
while extensions:
    base_name, extensions = os.path.splitext(base_name)

# Create time diff (send interval)
results['timestamps.client.send.diff'] = results['timestamps.client.send.wall'].diff()
average_time_diff = results['timestamps.client.send.diff'].mean()
std_dev_time_diff = results['timestamps.client.send.diff'].std()
print(f"Average send time difference: {average_time_diff/1000000.0} ms")
print(f"Standard send deviation of time difference: {std_dev_time_diff/1000000.0} ms")

# Append packet size
results['packet_length'] = latency_config['packet_length']

# Append arbitrary
if arguments_dict:
    for key in arguments_dict:
        results[key] = arguments_dict[key]

parquetfile = base_name + '.parquet'
print(f"Save results into {parquetfile}")

print(results.head(n=10).to_string(index=False))
results.to_parquet(parquetfile)

