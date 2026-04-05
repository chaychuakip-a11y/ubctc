#!/bin/bash
./selecttail wav mlf_sy mlf_fa_ph | ./raw_fea config.fea.16K_offCMN_PowerFB40 fb40 | ./selecttail mlf_sy fb40 mlf_fa_ph | ./randname
