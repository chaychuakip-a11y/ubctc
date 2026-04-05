#### generate mapping.txt（本地识别资源序列化使用） && 声学acmod.dnn.bin打包（atom_FA使用)

dir_hmm=`pwd`/final_s9k_for_ctc
dict=lib/kokr_20250221.dict

hmmlist=$dir_hmm/hmmlist.nosp.blank.final
models=$dir_hmm/MODELS.nosp.blank.txt
states_count=$dir_hmm/states.count.ctc-2.txt
syms_triphone=$dir_hmm/triphones_all.syms
out_mapping=$dir_hmm/mapping.txt

/work1/asrdictt/taoyu/sbin/CreatePhonesSymsFromDict.pl $dict $dir_hmm/phones_all.syms
/work1/asrdictt/taoyu/sbin/CreateTriphonesSymsFromHmmlist.pl $hmmlist $syms_triphone
/work1/asrdictt/taoyu/bin_esr/mapping.py $states_count $syms_triphone $hmmlist $models $out_mapping

/work1/asrdictt/taoyu/bin_atom/BuildAcModelTools $models $hmmlist $dir_hmm/atom_acmod.dnn.bin dnn $states_count
