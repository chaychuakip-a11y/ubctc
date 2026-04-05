#!/bin/bash
./selecttail wav mlf_sy mlf_fa_ph | ./wavAmplify_random wav out 0.3 0.05 | ./selecttail out mlf_sy mlf_fa_ph | ./renametail out wav | ./randname
