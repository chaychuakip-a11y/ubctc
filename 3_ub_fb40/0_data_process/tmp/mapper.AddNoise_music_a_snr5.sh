#!/bin/bash
./selecttail wav mlf_sy mlf_fa_ph | ./addtail seed.mlf randseed | ./AddNoise -n 2.700h_pure_music_data.pak -u -d -m snr_8khz -r 0 -s 5 -multiple 1 -output_type 2
