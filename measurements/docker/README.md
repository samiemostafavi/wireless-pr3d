# Create Dockerfile

Create and push
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
