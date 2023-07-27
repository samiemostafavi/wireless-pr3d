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

### Uplink

#### Measurement Session 1

**Advantech router 1**

Measured uplink bandwidth: 86.1Mbps

Run on the client container

- 85.44Mbps, 2x44.5kB packets, 120Hz, 8333 seconds (4x1666), 1e6 samples:
```
DEV_NAME=endnode01 NT_DEV=adv01 EXP_NAME=m1 SERVER_IP=10.70.70.3 ITNUM=4 SLEEP_DUR=60 TRIPM=oneway INTERVAL=8300us LENGTH=44500 MULT=2 DUR=1666s CL_PORT=55500 NT_SLEEP=300ms /tmp/measure-upload.sh > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 64kB packets, 120Hz, 8333 seconds (4x1666), 1e6 samples:
```
for i in `seq 1 4`; do (sleep 60 && irtt client --tripm=oneway -i 8300us -l 64000 -d 1666s -o d --outdir=/tmp/m1/client --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms /tmp/m1/networkinfo http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 2x42.6kB packets, 90Hz, 11111 seconds (7x1587), 1e6 samples:
```
for i in `seq 1 7`; do (sleep 60 && irtt client --tripm=oneway -i 11110us -l 42600 -m 2 -d 1587s -o d --outdir=/tmp/m1/client --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1587s 300ms /tmp/m1/networkinfo http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 2x64kB packets, 60Hz, 16666 seconds (10x1666), 1e6 samples:
```
for i in `seq 1 10`; do (sleep 60 && irtt client --tripm=oneway -i 16600us -l 64000 -m 2 -d 1666s -o d --outdir=/tmp/m1/client --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms /tmp/m1/networkinfo http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 4x64kB packets, 30Hz, 33333 seconds (20x1666), 1e6 samples:
```
for i in `seq 1 20`; do (sleep 60 && irtt client --tripm=oneway -i 33201us -l 64000 -m 4 -d 1666s -o d --outdir=/tmp/m1/endnode01/client --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms /tmp/m1/networkinfo http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 30.72Mbps, 32kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8300us -l 32000 -d 10s 10.70.70.3
```

- 30.72Mbps, 42.6kB packets, 90Hz:
```
irtt client --tripm=oneway -i 11110us -l 42600 -d 10s 10.70.70.3
```

- 30.72Mbps, 64kB packets, 60Hz:
```
irtt client --tripm=oneway -i 16600us -l 64000 -d 10s 10.70.70.3
```

- 30.72Mbps, 2x64kB packets, 30Hz:
```
irtt client --tripm=oneway -i 33200us -l 64000 -m 2 -d 10s 10.70.70.3
```

- 15.36Mbps, 16kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8300us -l 16000 -d 10s 10.70.70.3
```

- 15.36Mbps, 21.3kB packets, 90Hz:
```
irtt client --tripm=oneway -i 11110us -l 12000 -d 10s 10.70.70.3
```

- 15.36Mbps, 32kB packets, 60Hz:
```
irtt client --tripm=oneway -i 16600us -l 32000 -d 10s 10.70.70.3
```

- 15.36Mbps, 64kB packets, 30Hz:
```
irtt client --tripm=oneway -i 33200us -l 64000 -d 10s 10.70.70.3
```

- 7.68Mbps, 8kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8300us -l 8000 -d 10s 10.70.70.3
```

- 7.68Mbps, 10.65kB packets, 90Hz:
```
irtt client --tripm=oneway -i 11110us -l 10650 -d 10s 10.70.70.3
```

- 7.68Mbps, 16kB packets, 60Hz:
```
irtt client --tripm=oneway -i 16600us -l 16000 -d 10s 10.70.70.3
```

- 7.68Mbps, 32kB packets, 30Hz:
```
irtt client --tripm=oneway -i 33200us -l 32000 -d 10s 10.70.70.3
```

### Downlink 

On the edge container run:
```
irtt client --tripm=oneway -i 2ms -l 40000 -d 10s 172.16.0.8
```

## Upload and processing the files

On the endnode, upload the produced files located at client and networkinfo folders:
```
for file in /tmp/m1/client/*; do curl --user expeca:expeca --ftp-create-dirs -T ${file} ftp://10.70.70.3/mnt/volume/m1/endnode01/client/$(basename ${file}); done
```
```
for file in /tmp/m1/networkinfo/*; do curl --user expeca:expeca --ftp-create-dirs -T ${file} ftp://10.70.70.3/mnt/volume/m1/endnode01/networkinfo/$(basename ${file}); done
```

At the edge, run the following to create parquet files
```
python3 /tmp/parquets-from-folders.py /mnt/volume/m1/results /mnt/volume/m1/endnode01/client /mnt/volume/m1/edge/server /mnt/volume/m1/endnode01/networkinfo trip=uplink > /proc/1/fd/1 2>&1
```
