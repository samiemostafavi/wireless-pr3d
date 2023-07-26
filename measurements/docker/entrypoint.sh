#!/bin/bash

# run uFTP server
uFTP

# run measurement servers
if [[ -z "${SERVER_DIR}" ]]; then
  echo "SERVER_DIR env variable is not set"
  sh -c 'irtt server -i 0 -d 0 -l 0 -o d -q' & sh -c 'iperf3 -s' && fg
else
  echo "SERVER_DIR=${SERVER_DIR} env variable is set"
  mkdir -p $SERVER_DIR
  sh -c "irtt server -i 0 -d 0 -l 0 -o d --outdir=${SERVER_DIR} -q" & sh -c 'iperf3 -s' && fg
fi
