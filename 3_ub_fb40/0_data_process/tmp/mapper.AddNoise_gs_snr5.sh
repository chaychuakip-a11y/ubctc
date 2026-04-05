#!/bin/bash
./selecttail wav mlf_sy mlf_fa_ph | ./addtail seed.mlf randseed | ./AddNoise -n NoiseGS.addKTV.pcm.index.scp.tmp.pak.1 -u -d -m snr_8khz -r 0 -s 5 -multiple 1 -output_type 2
