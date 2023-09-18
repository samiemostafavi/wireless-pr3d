#!/bin/bash

# Example (uplink):
# DEV_NAME=endnode01 NT_DEV=adv01 EXP_NAME=m1 SERVER_IP=10.70.70.3 ITNUM=4 SLEEP_DUR=60 TRIPM=oneway INTERVAL=10ms INTERVAL_OFFSET=2ms LENGTH=44500 MULT=2 DUR=1666s CL_PORT=55500 NT_SLEEP=300ms /tmp/measure-upload.sh > /proc/1/fd/1 2>&1

# Example (downlink):
# DOWNLINK=true EDGE_NAME=edge END_NAME=endnode01 NT_DEV=adv01 EXP_NAME=m1 SERVER_IP=172.16.0.40 CLIENT_IP=10.70.70.3 ITNUM=4 SLEEP_DUR=60 TRIPM=oneway INTERVAL=10ms INTERVAL_OFFSET=2ms LENGTH=44500 MULT=2 DUR=1666s CL_PORT=55500 NT_SLEEP=300ms /tmp/measure-upload.sh > /proc/1/fd/1 2>&1


run_command_at_endnode() {
    local ip="$1"
    local command="$2"

    curl -s -X POST -H "Content-Type: application/json" -d "{\"cmd\": \"$command\"}" "http://$ip:50505/"
}

if [ "$DOWNLINK" = "true" ]; then
    echo "The DOWNLINK environment variable is set to true, measuring DOWNLINK."
    
    CL_OUT_DIR=/mnt/volume/$EXP_NAME/$EDGE_NAME/client/
    SE_OUT_DIR=/tmp/$EXP_NAME/$END_NAME/server/
    NT_OUT_DIR=/tmp/$EXP_NAME/networkinfo/
    SE_OUT_DIR_FIN=/mnt/volume/$EXP_NAME/$END_NAME/server/
    NT_OUT_DIR_FIN=/mnt/volume/$EXP_NAME/$END_NAME/networkinfo/
    for i in $(seq 1 $ITNUM); do
        (
            (sleep $SLEEP_DUR && irtt client --tripm=$TRIPM -i $INTERVAL -f $INTERVAL_OFFSET -g $EXP_NAME/$END_NAME -l $LENGTH -m $MULT -d $DUR -o d --outdir=$CL_OUT_DIR --local=:$CL_PORT $SERVER_IP) &
            (run_command_at_endnode "$SERVER_IP" "sleep $SLEEP_DUR && python3 /tmp/adv-mobile-info-recorder.py $DUR $NT_SLEEP $NT_OUT_DIR http://10.42.3.1:50500 device=$NT_DEV") &
            wait
            run_command_at_endnode "$SERVER_IP" "directory="$SE_OUT_DIR"; most_recent_file=\$(ls -t \$directory | grep -v '/$' | head -1); sefile=\$(readlink -f \$directory\$most_recent_file); curl --user expeca:expeca --ftp-create-dirs -T \${sefile} ftp://"$CLIENT_IP"/"$SE_OUT_DIR_FIN"/\$(basename \${sefile})"
            run_command_at_endnode "$SERVER_IP" "directory="$NT_OUT_DIR"; most_recent_file=\$(ls -t \$directory | grep -v '/$' | head -1); ntfile=\$(readlink -f \$directory\$most_recent_file); curl --user expeca:expeca --ftp-create-dirs -T \${ntfile} ftp://"$CLIENT_IP"/"$NT_OUT_DIR_FIN"/\$(basename \${ntfile})"
        )
    done
    
else
    echo "The DOWNLINK environment variable is not set to true, measuring UPLINK."

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

fi
