use strict;

my $dir_lib0       = "/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/1_down_pfile/lib_fb72";
my $dir_lib        = $dir_lib0; $dir_lib =~ s#/work1/#/yrfs4/#;
my $file_norm      = "$dir_lib0/fea.norm";
my $pfile_fea      = "$dir_lib/fea.pfile0";
my $pretraindata   = "$dir_lib/trainingbatchdata";

my $bin_pfile2pre  = "/work1/asrdictt/taoyu/tools/bin/pfileToPretraindata";

my $cmd;

mkdir $dir_lib if(!-e $dir_lib);

$cmd = "$bin_pfile2pre $pfile_fea $file_norm $pretraindata";

print $cmd."\n";
!system($cmd) or die;
system("touch $pretraindata.done");
