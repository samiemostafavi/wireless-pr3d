#!/bin/bash

if [[ -z "${WORKING_DIR}" ]]; then
  echo "WORKING_DIR env variable is not set"
else
  echo "WORKING_DIR=${WORKING_DIR} env variable is set, switching directory..."
  mkdir -p $WORKING_DIR
  cd $WORKING_DIR
fi
sh -c 'irtt server -i 0 -d 0 -l 0 -o d -q' & sh -c 'iperf3 -s' && fg
