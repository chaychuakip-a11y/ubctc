#!/bin/bash

required="MODELS.0.ori occ.0.ori hmmlist.context hmmlist.au HHEd.Clustering.ori HHEd.Addunseen HHEd.MixtureUp"
echo "DO Decision tree based state tying, $required come from MLE tcontext directory"
echo "NOTE: HHEd.MixtureUp can't be found in MLE tcontext directory, you can create it manually"
echo "      It is like this: "
echo "	    MU 4 { (sil).state[1-100].mix }"
echo "	    MU 4 { (sp).state[1-100].mix }"
echo "	    MU 2 { (*-*+*).state[1-100].mix }"

bin_hhed=/work1/asrdictt/taoyu/bin/hhed_ifly
dir_sbin_ctc=/work1/asrdictt/taoyu/sbin/for_ctc
dir_sbin=/work1/asrdictt/taoyu/sbin

hdir=/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/mle_ko_17kh/hmm/tcontext_s9k
hmmnum=9004;

dir_out=`pwd`/final_s9k_for_ctc

if [ ! -e $dir_out ]; then mkdir -p $dir_out; fi
if [ ! -e $dir_out/input ]; then mkdir -p $dir_out/input; fi

echo 'TI "sil_s2" { sil.state[2-4] }' >$dir_out/input/HHEd.TI_sp_sil
echo 'TI "sp_s2" { sp.state[2-4] }' >>$dir_out/input/HHEd.TI_sp_sil
echo 'MU 1 { (sil).state[1-100].mix }' >$dir_out/input/HHEd.MixtureUp
echo 'MU 1 { (sp).state[1-100].mix }' >>$dir_out/input/HHEd.MixtureUp
echo 'MU 1 { (*-*+*).state[1-100].mix }' >>$dir_out/input/HHEd.MixtureUp

hdfs dfs -copyToLocal $hdir/MODELS.0 $dir_out/input/MODELS.0.ori
hdfs dfs -copyToLocal $hdir/occ.0 $dir_out/input/occ.0.ori
hdfs dfs -copyToLocal $hdir/HHEd.Clustering $dir_out/input/HHEd.Clustering.ori
hdfs dfs -copyToLocal $hdir/HHEd.Addunseen $dir_out/input/
hdfs dfs -copyToLocal $hdir/hmmlist.au $dir_out/input/
hdfs dfs -copyToLocal $hdir/../context/hmmlist.context $dir_out/input/

for file in $required; do
	if [ ! -f $dir_out/input/$file ]; then
		"echo required file: $dir_out/input/$file"
		exit 1
	fi
done

cd $dir_out/input
$dir_sbin_ctc/ChangeHHEdClust.pl HHEd.Clustering.ori HHEd.Clustering $hmmnum
cat occ.0.ori | awk '{printf("%d %s %d %f\n",$1,$2,$3,$4+$5+$6)}' >occ.0

touch null
$bin_hhed -H MODELS.0.ori -w MODELS.0.txt null hmmlist.context
$dir_sbin_ctc/delModelState.pl MODELS.0.txt MODELS.0.txt.new
$bin_hhed -B -H MODELS.0.txt.new -w MODELS.0 null hmmlist.context

$bin_hhed -B -H MODELS.0 -w MODELS.1 HHEd.Clustering hmmlist.context
$bin_hhed -B -H MODELS.1 -w MODELS.2 HHEd.Addunseen hmmlist.tcontext
$bin_hhed -B -H MODELS.2 -w MODELS.final HHEd.MixtureUp hmmlist.final
$bin_hhed -H MODELS.final -w MODELS.txt HHEd.TI_sp_sil hmmlist.final
$dir_sbin/create_states_list.pl MODELS.txt states.list

mv {hmmlist.final,MODELS.final,MODELS.txt,TREE,states.list} $dir_out/
mv $dir_out/TREE $dir_out/TREE.final

cd $dir_out
$dir_sbin/create_states_list.pl MODELS.txt states.list
$dir_sbin_ctc/create_cdph_list.pl MODELS.txt states.list cdphone.list
$dir_sbin_ctc/create_cdph_map.pl hmmlist.final cdphone.list cdphone.map.list

$dir_sbin_ctc/model.del_sp.pl MODELS.txt MODELS.nosp.txt
grep -v 'sp' hmmlist.final >hmmlist.nosp.final
$dir_sbin/create_states_list.pl MODELS.nosp.txt states.list.nosp

$dir_sbin_ctc/model.sp2blank.pl MODELS.txt MODELS.nosp.blank.txt
cp -p hmmlist.nosp.final hmmlist.nosp.blank.final
echo 'blank' >>hmmlist.nosp.blank.final
$dir_sbin/create_states_count_for_ctc.pl states.list.nosp states.count.ctc-2.txt -2
