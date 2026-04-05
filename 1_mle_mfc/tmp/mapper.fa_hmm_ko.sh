#!/bin/bash
./raw_fea config.fea.16K_offCMN_PowerFB24_0_D_A fb72 | ./cmvn_simple 2 24 1 fb72 | ./raw_fea config.fea.16K_offCMN_PowerMFCC_0_D_A fea | ./cmvn_simple 2 13 1 fea | ./pakeditmap_htkfea -FA_State_Align MODELS hmmlist.final kokr_20250221.dict.align | ./selecttail fb72 mlf_sy mlf_fa_ph | ./randname
