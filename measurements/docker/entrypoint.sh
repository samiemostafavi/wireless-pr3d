#!/bin/bash

sh -c 'irtt server -i 0 -d 0 -l 0 -o d -q' & sh -c 'iperf3 -s' && fg
