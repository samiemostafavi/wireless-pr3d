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

- 61.44Mbps, 64kB packets, 120Hz, 8333 seconds (4x1666), 1e6 samples:
```
cd /mnt/client/m1; for i in `seq 1 4`; do (sleep 60 && irtt client --tripm=oneway -i 8300us -l 64000 -d 1666s -o d --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 2x42.6kB packets, 90Hz, 11111 seconds (7x1587), 1e6 samples:
```
cd /mnt/client/m1; for i in `seq 1 7`; do (sleep 60 && irtt client --tripm=oneway -i 11110us -l 42600 -m 2 -d 1587s -o d --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1587s 300ms http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 2x64kB packets, 60Hz, 16666 seconds (10x1666), 1e6 samples:
```
cd /mnt/client/m1; for i in `seq 1 10`; do (sleep 60 && irtt client --tripm=oneway -i 16600us -l 64000 -m 2 -d 1666s -o d --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
```

- 61.44Mbps, 4x64kB packets, 30Hz, 33333 seconds (20x1666), 1e6 samples:
```
cd /mnt/client/m1; for i in `seq 1 20`; do (sleep 60 && irtt client --tripm=oneway -i 33201us -l 64000 -m 4 -d 1666s -o d --local=:55500 10.70.70.3) & (sleep 60 && python3 /tmp/adv-mobile-info-recorder.py 1666s 300ms http://10.42.3.1:50500 device=adv01) & wait; done > /proc/1/fd/1 2>&1
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
