####声学acmod.dnn.bin打包
dir_hmm=`pwd`/final_s9k
dict=lib/kokr_20250221.dict

if [ ! -e $dir_hmm/states.count.fake.txt ]; then
HHEd -H $dir_hmm/MODELS -w $dir_hmm/MODELS.txt /work1/asrdictt/taoyu/bin/null $dir_hmm/hmmlist.final
/work1/asrdictt/taoyu/sbin/create_states_list.pl $dir_hmm/MODELS.txt $dir_hmm/states.list
/work1/asrdictt/taoyu/sbin/create_states_count_for_ctc.pl $dir_hmm/states.list $dir_hmm/states.count.fake.txt
fi
/work1/asrdictt/taoyu/bin_atom/BuildAcModelTools $dir_hmm/MODELS.txt $dir_hmm/hmmlist.final $dir_hmm/atom_acmod.dnn.bin dnn $dir_hmm/states.count.fake.txt

/work1/asrdictt/taoyu/sbin/CreatePhonesSymsFromDict.pl $dict $dir_hmm/phones_all.syms
/work1/asrdictt/taoyu/sbin/CreateTriphonesSymsFromHmmlist.pl $dir_hmm/hmmlist.final $dir_hmm/triphones_all.syms
echo [common] >$dir_hmm/MappingBuilder.cfg
echo StateFile        = $dir_hmm/states.count.fake.txt >>$dir_hmm/MappingBuilder.cfg
echo TriphonesFile    = $dir_hmm/triphones_all.syms >>$dir_hmm/MappingBuilder.cfg
echo HMMListFinalFile = $dir_hmm/hmmlist.final >>$dir_hmm/MappingBuilder.cfg
echo ModelsTxtFile    = $dir_hmm/MODELS.txt >>$dir_hmm/MappingBuilder.cfg
echo OutputMatFile    = $dir_hmm/Mapping.wts >>$dir_hmm/MappingBuilder.cfg

/work1/asrdictt/taoyu/bin_atom/MappingBuilder $dir_hmm/MappingBuilder.cfg
