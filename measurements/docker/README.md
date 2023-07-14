# Network Measurement Service

Create Docker image and push
```
docker build -t samiemostafavi/perf-meas .
docker image push samiemostafavi/perf-meas
```

To test it, run this on all nodes
```
docker run -d --name perf-meas --net=host samiemostafavi/perf-meas
```

You can change the location that irtt saves log files by changing the working directory
```
docker run -d --name perf-meas --net=host -e WORKING_DIR=/home/ samiemostafavi/perf-meas
```

To measure bandwidth
```
docker exec -it perf-meas iperf3 -c <address> -u -b 1G --get-server-output
```

To measure latency
```
docker exec -it perf-meas irtt client --tripm=round -i 5ms -l 10000 -d 10s <address>
```

# Advantech router mobile network info

Follow instructions [here](https://github.com/samiemostafavi/advmobileinfo)

To test it, run `curl http://10.10.5.1:50500` from another machine. The result is a JSON:
```
{"Registration": "Home Network", "Operator": "999 08", "Technology": "NR5G", "PLMN": "99908", "Cell": "7534001", "LAC": "0BC2", "Channel": "650688", "Band": "n78", "Signal Strength": "-70 dBm", "Signal Quality": "-11 dB", "RSRP": "-70 dBm", "RSRQ": "-11 dB", "CSQ": "21"}
```

Inside perf-meas container, you can run
```
python3 /tmp/adv-mobile-info-recorder.py 10s 100ms http://10.10.5.1:50000 adv01ul
```

