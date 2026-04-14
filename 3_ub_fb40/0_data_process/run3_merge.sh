#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting final merge process..."
perl 9_fea_merge.pl > $LOG_DIR/9_fea_merge.log 2>&1

if [ $? -eq 0 ]; then
    echo "Merge finished. Check log in $LOG_DIR/9_fea_merge.log"
else
    echo "Error in 9_fea_merge.pl. Check $LOG_DIR/9_fea_merge.log"
    exit 1
fi
