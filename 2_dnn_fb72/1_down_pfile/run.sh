nSplit=10
dir_lib=${PWD/work1/yrfs4}/lib_fb72

wait_dir_hdfs.pl /workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_fb72_fa_state/_SUCCESS

for ((i=0;i<$nSplit;i++)); do ky exp submit CommonJob -n data -d get-pfile-$i -e "1_get_pfile_from_hdfs.pl $i" -l 1_get_pfile_from_hdfs.$i.log -i reg.deeplearning.cn/dlaas/cv_dist_openmpi:0.1 -c 8 -m 20 -x 'debug-12-146,debug-12-147' -r dlp3-asrdictt-car-reserved; sleep 4; done;
for ((i=0;i<$nSplit;i++)); do wait_file.pl ${dir_lib}/fea.norm$i.done; done;

perl 2_create_fea_norm_with_pfile.pl
perl 2_create_states_count_with_pfile.pl

ky exp submit CommonJob -n data -d pfile2pretraindata -e "3_pfile2pretraindata.pl" -l 3_pfile2pretraindata.log -i reg.deeplearning.cn/dlaas/cv_dist_openmpi:0.1 -c 8 -m 20 -x 'debug-12-146,debug-12-147' -r dlp3-asrdictt-car-reserved
