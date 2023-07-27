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
docker run -d --name perf-meas --net=host -e SERVER_DIR=/home/ samiemostafavi/perf-meas
```

To measure bandwidth
```
docker exec -it perf-meas iperf3 -c <address> -u -b 1G --get-server-output
```

To measure latency
```
docker exec -it perf-meas irtt client --tripm=round -i 5ms -l 10000 -d 10s <address>
```

# Download a file from the container

We run a tiny FTP server inside the contrainer: [uFTP](https://www.uftpserver.com/wiki/uftp-server-installation)

You can download any files on the container with `expeca` as username and password
```
wget ftp://expeca:expeca@192.168.2.2/tmp/entrypoint.sh
```

Or download a directory
```
wget -r ftp://expeca:expeca@192.168.2.2/tmp/
```

Upload one file to the edge container:
```
curl -T cl_10-42-3-2_55500_20230726_142750.json.gz ftp://10.70.70.3 --user expeca:expeca
```

Upload a directory:
```
for file in /tmp/m1/client/*; do curl --user expeca:expeca --ftp-create-dirs -T ${file} ftp://10.70.70.3/mnt/volume/m1/edgenode01/client/$(basename ${full_name}); done
```

# Advantech router mobile network info

Follow instructions [here](https://github.com/samiemostafavi/advmobileinfo)

To test it, run `curl http://10.10.5.1:50500` from another machine. The result is a JSON:
```
{"Registration": "Home Network", "Operator": "999 08", "Technology": "NR5G", "PLMN": "99908", "Cell": "7534001", "LAC": "0BC2", "Channel": "650688", "Band": "n78", "Signal Strength": "-70 dBm", "Signal Quality": "-11 dB", "RSRP": "-70 dBm", "RSRQ": "-11 dB", "CSQ": "21"}
```

Inside perf-meas container, you can run
```
python3 /tmp/adv-mobile-info-recorder.py 10s 100ms /mnt/client/m1/networkinfo http://10.10.5.1:50500 device=adv01
```

# Make Parquet files

Use this command to combine latency and network files and convert them to Parquet using Python script
```
python3 /tmp/makeparquet.py /mnt/client/m1/client/cl_104232_55500_2023725_15730.json.gz /mnt/server/m1/server/se_1721608_55500_2023725_15730.json.gz /mnt/client/m1/network/mni_20230725_150725.json.gz trip=uplink
```

Use this command to combine files on the client side and server side and make a parquet
```
python3 parquets-from-folders.py testfiles/results testfiles/client testfiles/server testfiles/networkinfo trip=uplink
```

# Upload files to Swift

First fix the nameserver on the container
```
echo nameserver 8.8.8.8 > /etc/resolv.conf
```

Use this command to upload file `/mnt/client/m1/networkinfo/adv01ul_20230718_173430.json.gz` to `m1` container in Swift.
```
AUTH_SERVER=testbed.expeca.proj.kth.se AUTH_PROJECT_NAME=sdr-test-project AUTH_USERNAME=samie AUTH_PASSWORD=password python3 /tmp/upload-files.py /mnt/client/m1/networkinfo/adv01ul_20230718_173430.json.gz m1
```

Multiple `json.gz` files in folder `/mnt/client/m1/networkinfo/`:
```
for f in /mnt/client/m1/networkinfo/*.json.gz; do AUTH_SERVER=testbed.expeca.proj.kth.se AUTH_PROJECT_NAME=sdr-test-project AUTH_USERNAME=samie AUTH_PASSWORD=password python3 /tmp/upload-files.py $f m1; done
```
