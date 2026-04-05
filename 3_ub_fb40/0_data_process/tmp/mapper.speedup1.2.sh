#!/bin/bash
./selecttail wav mlf_sy | ./wav_speed_hadoop -speech -tempo=20% | ./selecttail out mlf_sy | ./renametail out wav | ./randname
