#!/bin/bash

if [ $# -ne 1 ] ; then
    echo "Usage: train.sh devices(-1:means running in dlp)"
    echo "For example1: train.sh -1"
    echo "For example2: train.sh 2,3"
    exit 1
fi

devices=$1
ngpu=$(echo $devices | tr ',' '\n' | wc -l)

if [ "$devices" != "-1" ] ; then #### if running in local machine, then set gpu id;
export NWORKER=1
export NGPU_PER_WORKER=$ngpu
export CUDA_VISIBLE_DEVICES=$devices
else
ngpu=-1
fi

source /home3/asrdictt/taoyu/bashrc_mxnet

echo ngpu:$ngpu
echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES
echo NWORKER:$NWORKER
echo NGPU_PER_WORKER=$NGPU_PER_WORKER
echo `which python`
echo $PATH
echo $LD_LIBRARY_PATH

if [ -f stop ]; then /bin/rm -rf stop; fi

count=0
while [ ! -f stop ]; do
    perl train.pl $ngpu >train.${count}.log 2>&1
    sleep 60
    count=$((count + 1))
    echo "MPI run died, restart $count"
done
