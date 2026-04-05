#!/bin/bash
./selecttail wav mlf_sy mlf_fa_ph | ./addtail seed.mlf randseed | ./AddNoise -n iflytek-20180521-part-00000 -u -d -m snr_8khz -r 0 -s 5 -multiple 1 -output_type 2
