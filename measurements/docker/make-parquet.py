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
# python3 make-parquet.py 172-16-0-8-36970_7-18-23-26-32.json.gz adv01ul_20230718_232628.json.gz

# Open gzip file storing a json
def opengzip(filename):
    with gzip.open(filename, 'r') as fin:            # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
    
    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

latencyfile = str(sys.argv[1]) # Input latency json.gz file
netinfofile = str(sys.argv[2]) # Input network information json.gz file
  
# read gz files into dict
latencydata = opengzip(latencyfile)
print(f"Latency file contains {len(latencydata)} records")
netinfodata = opengzip(netinfofile)
print(f"Network info file contains {len(netinfodata)} records")

# preprocess netinfo
for i in range(len(netinfodata)):
    if not netinfodata[i]['Band']:
        del netinfodata[i]

if "oneway_trips" in latencydata:
    print("Processing: oneway trips")

    latency_samples = latencydata["oneway_trips"]
    
elif "round_trips" in latencydata:
    print("Processing: round trips")
    
    latency_samples = latencydata["round_trips"]

else:
    print("corrupt latency json")
    exit(0)

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

parquetfile = base_name + '.parquet'
print(f"Save results into {parquetfile}")
results.to_parquet(parquetfile)

