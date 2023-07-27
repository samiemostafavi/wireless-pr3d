# Measurement Schemes

We create predictors with the following conditions:
- Load parameters
- Channel quality (CSQ, RSRQ, RSRP)

We test whether the prediction is accurate:
- Uplink and downlink at the same time?
- Try somoe other times
- Try another device with the same channel quality

NOTES:
- `sleep 60` is important to avoid errors. After each measurement round, irtt server needs to save the data, this takes time.
- Also, if the client does not change its port every time it is better. That is why `55500` is set as the port.
- `> /proc/1/fd/1 2>&1` sends stdout and stderr to the container's main logs terminal.

## Measurement Session 1 

- Group name: `m1`
- Device: `adv01`
- Direction: `uplink`
- Maximum throughput: `86.1Mbps`

Latency measurements:

- For 85.44Mbps, 2x44.5kB packets, 120Hz, 8333 seconds (4x1666), 1e6 samples run at the end-node container:
```
DEV_NAME=endnode01 NT_DEV=adv01 EXP_NAME=m1 SERVER_IP=10.70.70.3 ITNUM=4 SLEEP_DUR=60 TRIPM=oneway INTERVAL=8300us LENGTH=44500 MULT=2 DUR=1666s CL_PORT=55500 NT_SLEEP=300ms /tmp/measure-upload.sh > /proc/1/fd/1 2>&1
```

- At the edge, run the following to create parquet files
```
python3 /tmp/parquets-from-folders.py /mnt/volume/m1/results /mnt/volume/m1/endnode01/client /mnt/volume/m1/edge/server /mnt/volume/m1/endnode01/networkinfo trip=uplink > /proc/1/fd/1 2>&1
```
- Create container `m1` on Openstack object store service.
- On the edge container, run the following to upload the files there (replace username and password):
```
for f in /mnt/volume/m1/results/*.parquet; do AUTH_SERVER=testbed.expeca.proj.kth.se AUTH_PROJECT_NAME=sdr-test-project AUTH_USERNAME=username AUTH_PASSWORD=password python3 /tmp/upload-files.py $f m1; done
```
