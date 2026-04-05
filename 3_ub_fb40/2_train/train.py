from asr.train import train, Train
import time

import os,sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: py config.ini')
    config = sys.argv[1]
    print(config)
    mytrain = Train()
    mytrain.load_config(config)
    mytrain.start_train()

    print("----please restart----")
    print("----sleep 10s----",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("----begin restart----",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
