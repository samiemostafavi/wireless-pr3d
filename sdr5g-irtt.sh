#!/bin/bash

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_7.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_8.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_9.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_10.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_11.json --fill=rand 12.1.1.1
