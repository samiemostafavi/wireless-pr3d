import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import gzip
from pathlib import Path
import sys

latencyfile = str(sys.argv[1])
netinfofile = str(sys.argv[2])

# Open gzip file storing a json
def opengzip(filename):
    with gzip.open(filename, 'r') as fin:            # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
    
    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data
  
# read gz files into dict
latencydata = opengzip(latencyfile)
netinfodata = opengzip(netinfofile)

round_trips = latencydata["round_trips"]

# Lists of delays
data_rtt_list = [round_trip['delay']['rtt'] for round_trip in round_trips] # list of rtt values
data_rx_list = [round_trip['delay']['receive'] for round_trip in round_trips] # list of rx times
data_tx_list = [round_trip['delay']['send'] for round_trip in round_trips] # list of tx times
data_timestamps_list = [round_trip['timestamps']['client']['send']['wall'] for round_trip in round_trips] # list of timestamps

L = len(round_trips)
X_list = [X] * L # create a list with X value and same elements as round_trips
Y_list = [Y] * L # create a list with Y value and same elements as round_trips

# Lists of network conditions
data_ntw_cond_timestamps = [measurement['timestamp'] for measurement in data2]
data_RSRP = [measurement['RSRP'] for measurement in data2]
data_RSRQ = [measurement['RSRQ'] for measurement in data2]
data_channel = [measurement['Channel'] for measurement in data2]
data_band = [measurement['Band'] for measurement in data2]

ntw_cond_t = np.array(data_ntw_cond_timestamps)
idx = np.array([np.argmin(abs(timestamp-ntw_cond_t)) for timestamp in data_timestamps_list])

data_df = pd.DataFrame({'rtt':data_rtt_list, 
                        'send':data_tx_list, 
                        'receive':data_rx_list, 
                        'timestamp':data_timestamps_list, 
                        'X':X_list, 
                        'Y':Y_list,
                        'RSRP':[data_RSRP[i] for i in idx],
                        'RSRQ':[data_RSRQ[i] for i in idx],
                        'channel':[data_channel[i] for i in idx],
                        'band':[data_band[i] for i in idx]})

print(data_df.head(10))

table_data_rtt = pa.Table.from_pandas(data_df) # create table
pq.write_table(table_data_rtt, 'dataset_val2_'+num+'.parquet') # create parquet file 
