#!/bin/bash

sh -c 'irtt server -i 1ms -o d -q' & sh -c 'iperf3 -s' && fg
