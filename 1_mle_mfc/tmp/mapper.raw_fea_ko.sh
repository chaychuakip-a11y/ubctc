#!/bin/bash
./raw_fea config.fea.16K_offCMN_PowerMFCC_0_D_A fea | ./cmvn_simple 2 13 1 fea | ./selecttail fea mlf_sy
