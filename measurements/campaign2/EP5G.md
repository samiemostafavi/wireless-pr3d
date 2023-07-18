# ExPECA Testbed EP5G Setup

Reserve
* EP5G
* Advantech-01
* Worker-01

## Bring Up the edge server on Worker-01

Networks:
* Edge-Net (`10.70.70.0/24`)

ENV Variables
```
None
```

Labels
```
networks.1.interface=eno12419,networks.1.ip=10.70.70.3/24,networks.1.routes=172.16.0.0/16-10.70.70.1
```

Test:
Make sure from Advantech-01 you can ping `10.70.70.3`.


## Bring up the client on Worker-01

Networks:
* Adv-01-net (`10.42.3.0/24`)

ENV Variables
```
None
```

Labels
```
networks.1.interface=eno12429,networks.1.ip=10.42.3.2/24,networks.1.routes=10.70.70.0/24-10.42.3.1
```

Test
Ping `10.70.70.3` from the container.

## Measure available bandwidth

**Uplink)** On the client container run:
```
iperf3 -c 10.70.70.3 -u -b 1G --get-server-output
```

**Downlink)** On the edge container run:
```
iperf3 -c 172.16.0.8 -u -b 1G --get-server-output
```

## Measure latency

**Uplink)** On the client container

- 64Mbps, 64kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8300us -l 64000 -d 30s 10.70.70.3
```

- 32Mbps, 64kB packets, 60Hz:
```
irtt client --tripm=oneway -i 16600us -l 64000 -d 10s 10.70.70.3
```

- 32Mbps, 32kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8300us -l 32000 -d 10s 10.70.70.3
```

- 16Mbps, 16kB packets, 120Hz:
```
irtt client --tripm=oneway -i 8ms -l 16000 -d 10s 10.70.70.3
```

- 16Mbps, 32kB packets, 60Hz:
```
irtt client --tripm=oneway -i 16ms -l 20000 -d 10s 10.70.70.3
```

- 16Mbps, 40kB packets, 30Hz:
```
irtt client --tripm=oneway -i 32ms -l 40000 -d 10s 10.70.70.3
```

**Downlink)** On the edge container run:
```
irtt client --tripm=oneway -i 2ms -l 40000 -d 10s 172.16.0.8
```

## Measure Advantech router mobile info

Run this on the client container:
```
python3 /tmp/adv-mobile-info-recorder.py 10s 100ms http://10.42.3.1:50500 adv01ul
```
