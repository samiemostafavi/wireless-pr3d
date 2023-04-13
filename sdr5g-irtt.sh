#!/bin/bash

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_5.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_6.json --fill=rand 12.1.1.1
sleep 30

irtt client -i 10ms -d 15m -l 75 -o /home/wlab/irtt_data/sdr5g/rtts_7.json --fill=rand 12.1.1.1
