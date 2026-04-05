#!/bin/bash
./selecttail mlf_sy wav | ./mt_mae jiashi_open.txt 1.0 noise_L.scp.tmp.pak.1 noise_R.scp.tmp.pak.1 | ./selecttail wav mlf_sy | ./randname
