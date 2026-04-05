# mkdir lib
# /work1/asrdictt/taoyu/sbin/dict/CheckDictWithHmmlist.pl ../../res/kokr_20250221.dict ../../res/hmmlist.mono
# dict.create_for_mle.py ../../res/kokr_20250221.dict lib/kokr_20250221.dict
# hmmlist.to_modelgen.py ../../res/hmmlist.mono lib/modelgen.ko.s3 5

perl 1_getfeature.mfcc.pl
perl 2_HadoopMLE-v2.pl GV-MLE-v2.pm
sh 3_pack_hmm.sh
perl 4_fa_state.mle.pl

sh 5_1_ctc_cdphone.sh
sh 5_2_ctc_pack_res.sh
