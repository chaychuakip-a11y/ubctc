use strict;

use lib "/work1/asrdictt/taoyu/sbin";
use share_hadoop;

my $jobqueue     = "nlp";
my $jobname      = "raw_fea_ko";
my $num_reduce   = 0;
my $in_blocksize = 512*1024*1024;
my $block_size   = 64*1024*1024;
my $replication  = 2;

my @hdir_src     = (
                   "/workdir/asrdictt/dasrdictt/taoyu/mlg/korean/korean_tx_6.7kh",   ### 7000H
                   "/workdir/asrdictt/dasrdictt/taoyu/mlg/korean/korean_zx_11.4kh",  ### 4000H
                   "/workdir/asrdictt/dasrdictt/taoyu/mlg/korean/korean_lyb_490h",
                   "/workdir/asrdictt/dasrdictt/taoyu/mlg/korean/korean_kaiyuan_tx_32.9kh/*part-000*",     ### 100/800,  4000H
                   "/workdir/asrdictt/dasrdictt/taoyu/mlg/korean/korean_kaiyuan_zx_6.5kh/*part-000[0-1]?", ### 2/8, 1000H
);
my $hdir_out     = "/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_mfcc";
my $hdir_src     = join(" -input ", @hdir_src);

my $dir_bin      = "/work1/asrdictt/taoyu/bin";

my $config1      = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config2      = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A";

my $bin_raw_fea  = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $bin_cmvn     = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $bin_hvad     = "$dir_bin/htk-0.1.4/bin/hvad";
my $bin_w_vad_so = "$dir_bin/htk-0.1.4/bin/w_vad.so";

my $bin_selecttail  = "$dir_bin/selecttail";
my $bin_randname    = "$dir_bin/randname";
my $bin_randnamered = "$dir_bin/randnamered";

my $bin_stream   = "$dir_bin/streamingAC-2.5.0.jar";

my $dir_tmp = 'tmp'; mkdir $dir_tmp if !-e $dir_tmp;
my @cmd_map;
my @cmd_red;
my @files;

my $cmd_map;
my $cmd_red;
my $cmd;
my $files;

@cmd_map = (
# ::CmdToLocal("$bin_raw_fea $config1 fb72"),
# ::CmdToLocal("$bin_cmvn 2 24 1 fb72"),
::CmdToLocal("$bin_raw_fea $config2 fea"),
::CmdToLocal("$bin_cmvn 2 13 1 fea"),
::CmdToLocal("$bin_selecttail fea mlf_sy"),
#::CmdToLocal("$bin_randname"),
);

@cmd_red = (
#::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_raw_fea,
$config1,
$config2,
$bin_cmvn,
$bin_selecttail,
$bin_randname,
$bin_randnamered,
);

$cmd_map = join(" | ", @cmd_map);
$cmd_red = join(" | ", @cmd_red);

if(@cmd_map > 1)
{
    open(OUT, ">", "$dir_tmp/mapper.$jobname.sh") || die $!;
    print OUT "#!/bin/bash\n";
    print OUT "$cmd_map\n";
    close OUT;
    push(@files, "$dir_tmp/mapper.$jobname.sh");
    $cmd_map = "bash ./mapper.$jobname.sh";
}

if(@cmd_red > 1)
{
    open(OUT, ">", "$dir_tmp/reducer.$jobname.sh") || die $!;
    print OUT "#!/bin/bash\n";
    print OUT "$cmd_red\n";
    close OUT;
    push(@files, "$dir_tmp/reducer.$jobname.sh");
    $cmd_red = "bash ./reducer.$jobname.sh";
}

$files = join(",", @files);

$cmd = "hadoop jar $bin_stream "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=2000 "
."-Dmapreduce.reduce.memory.mb=1000 "
."-Ddc.input.block.size=$in_blocksize "
."-Ddfs.blocksize=$block_size "
."-Ddfs.replication=$replication "
."-files \"$files\" "
."-input $hdir_src "
."-output $hdir_out "
;

$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

::RemoveHadoopDirIfExist($hdir_out);
::PR($cmd);
::SuccessOrDie("$hdir_out");
