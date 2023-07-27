#!/bin/bash

# Example:
# DEV_NAME=endnode01 NT_DEV=adv01 EXP_NAME=m1 SERVER_IP=10.70.70.3 ITNUM=4 SLEEP_DUR=60 TRIPM=oneway INTERVAL=8300us LENGTH=44500 MULT=2 DUR=1666s CL_PORT=55500 NT_SLEEP=300ms /tmp/measure-upload.sh > /proc/1/fd/1 2>&1

CL_OUT_DIR=/tmp/$EXP_NAME/client/
NT_OUT_DIR=/tmp/$EXP_NAME/networkinfo/
for i in $(seq 1 $ITNUM); do
    (
        (sleep $SLEEP_DUR && irtt client --tripm=$TRIPM -i $INTERVAL -g $EXP_NAME/edge -l $LENGTH -m $MULT -d $DUR -o d --outdir=$CL_OUT_DIR --local=:$CL_PORT $SERVER_IP) &
        (sleep $SLEEP_DUR && python3 /tmp/adv-mobile-info-recorder.py $DUR $NT_SLEEP $NT_OUT_DIR http://10.42.3.1:50500 device=$NT_DEV) &
        wait
        clfile=$(find "$CL_OUT_DIR" -maxdepth 1 -type f -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
        ntfile=$(find "$NT_OUT_DIR" -maxdepth 1 -type f -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
        curl --user expeca:expeca --ftp-create-dirs -T ${clfile} ftp://$SERVER_IP/mnt/volume/$EXP_NAME/$DEV_NAME/client/$(basename ${clfile})
        curl --user expeca:expeca --ftp-create-dirs -T ${ntfile} ftp://$SERVER_IP/mnt/volume/$EXP_NAME/$DEV_NAME/networkinfo/$(basename ${ntfile})
    )
done
