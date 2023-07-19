# ExPECA Testbed EP5G Setup

Reserve
* EP5G
* Advantech-01
* Worker-01

Create 2 volumes, one for the server and one for the client.
* server-volume
* client-volume

## Bring Up the edge server on Worker-01

Networks:
* Edge-Net (`10.70.70.0/24`)

Volumes:
* server-volume
* mount on `/mnt/server`

ENV Variables
```
WORKING_DIR=/mnt/server/m1
```

Labels
```
networks.1.interface=eno12419,networks.1.ip=10.70.70.3/24,networks.1.routes=172.16.0.0/16-10.70.70.1
```

Test:
Make sure from Advantech-01 you can ping `10.70.70.3`.

Also, type `cd /mnt/server/m1` before any command to change the working directory.

## Bring up the client on Worker-01

Networks:
* Adv-01-net (`10.42.3.0/24`)

Volumes:
* client-volume
* mount on `/mnt/client`

ENV Variables
```
WORKING_DIR=/mnt/client/m1
```

Labels
```
networks.1.interface=eno12429,networks.1.ip=10.42.3.2/24,networks.1.routes=10.70.70.0/24-10.42.3.1
```

Test
Ping `10.70.70.3` from the container.

Also, type `cd /mnt/client/m1` before any command to change the working directory.

### Measure available bandwidth (test)

**Uplink)** On the client container run:
```
iperf3 -c 10.70.70.3 -u -b 1G --get-server-output
```

**Downlink)** On the edge container run:
```
iperf3 -c 172.16.0.8 -u -b 1G --get-server-output
```

### Measure latency (test)

**Uplink)** On the client container

- 61.44Mbps, 64kB packets, 120Hz:
```
irtt client --tripm=oneway -i 2ms -l 64000 -d 10s 10.70.70.3
```

**Downlink)** On the edge container run:
```
irtt client --tripm=oneway -i 2ms -l 64000 -d 10s 172.16.0.8
```

### Measure Advantech router mobile info (test)

Run this on the client container:
```
python3 /tmp/adv-mobile-info-recorder.py 10s 100ms http://10.42.3.1:50500 adv01ul
```

Measure latency with mobile info:
```
irtt client --tripm=oneway -i 2ms -l 64000 -d 10s 10.70.70.3 & python3 /tmp/adv-mobile-info-recorder.py 10s 300ms http://10.42.3.1:50500 adv01ul && fg
```
