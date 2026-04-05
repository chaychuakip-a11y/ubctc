nSplit=10
dir_lib=${PWD/work1/yrfs4}/lib_fb40
dir_lmdb=${PWD/work1/yrfs4}/lmdb_fb40

# for ((i=0;i<$nSplit;i++)); do ky exp submit CommonJob -n data-pfile -d get-pfile-$i -e "1_get_pfile_from_hdfs.pl $i" -l 1_get_pfile_from_hdfs.$i.log -i reg.deeplearning.cn/dlaas/cv_dist_openmpi:0.1 -c 8 -m 20 -r dlp3-asrdictt-car-reserved; sleep 1; done;
# for ((i=0;i<$nSplit;i++)); do /work1/asrdictt/taoyu/sbin/wait_file.pl ${dir_lib}/fea.norm$i.done; done;

# perl 2_create_fea_norm_with_pfile.pl
# perl 2_create_states_count_with_pfile.pl

for ((i=0;i<$nSplit;i++)); do ky exp submit CommonJob -n data-pfile2lmdb -d pfile2lmdb-$i -e "3_pfile2lmdb.pl $i" -l 3_pfile2lmdb.$i.log -i reg.deeplearning.cn/dlaas/cv_dist_openmpi:0.1 -c 8 -m 20 -r dlp3-asrdictt-car-reserved; sleep 1; done;
for ((i=0;i<$nSplit;i++)); do /work1/asrdictt/taoyu/sbin/wait_file.pl ${dir_lmdb}/train_part$i/done; done;
