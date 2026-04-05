#!/opt/tool/anaconda3.5/bin/python3
#coding=utf-8

import os,sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: py ngpu')
    ngpu = int(sys.argv[1])
    
    if os.path.exists('stop'):
        os.remove('stop')
    
    count = 0
    while not os.path.exists('stop'):
        # cmdline = 'perl train_init.pl -1'
        cmdline = 'perl train.pl {} >train.{}.log 2>&1'.format(ngpu, count)
        os.system(cmdline)
        time.sleep(60)
        count += 1
        print('MPI run died, restart ', count)
