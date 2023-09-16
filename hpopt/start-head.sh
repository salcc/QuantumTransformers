#!/bin/bash
# Script based on https://github.com/NERSC/slurm-ray-cluster/blob/master/start-head.sh

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray head node"
# Launch the head node
ray start --head --node-ip-address=$1 --port=6379
sleep infinity