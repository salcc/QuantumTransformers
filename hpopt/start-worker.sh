#!/bin/bash
# Script based on https://github.com/NERSC/slurm-ray-cluster/blob/master/start-worker.sh

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node"
ray start --address $1
sleep infinity