#!/bin/bash
./raw_fea config.fea.16K_offCMN_PowerFB24_0_D_A fb72 | ./cmvn_simple 2 24 1 fb72 | ./atom -c atom_fa.dnnfa_maeclose.cfg -mtn 2 -dtn 1 | ./selecttail wav mlf_sy mlf_fa_ph | ./randname
