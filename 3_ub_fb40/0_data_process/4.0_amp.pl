
use strict;

require "./utils.pl";
my $config_data;
BEGIN {
    $config_data = load_config();
}

use lib $config_data->{dir_sbin};
use share_hadoop;

my $jobname           = "amp";#SET
my $jobqueue          = $config_data->{jobqueue} || "nlp";#SET
my $num_reduce        = 10;
my $in_blocksize      = 512*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;


my @hdir_src; if (@ARGV > 0) {
    @hdir_src = ($ARGV[0]);
} else {
    @hdir_src = (
                    "$config_data->{hdfs_out_root}/wav_dnnfa/*-part-000[2-3]?",
                   );
}
my $hdir_out     = ("$config_data->{hdfs_out_root}/wav_amp_0.2.wav_dnnfa");
my $hdir_src     = join(" -input ", @hdir_src);
my $dir_tmp      = "tmp"; mkdir $dir_tmp if !-e $dir_tmp;
foreach my $hdir_cur (@hdir_src)
{
	if($hdir_cur =~ /\*/ && $hdir_cur =~ /part/)
	{
		$hdir_cur =~ s#/[^/]+$##;
	}
	$hdir_cur .= '/_SUCCESS';
	system("$config_data->{dir_sbin}/wait_dir_hdfs.pl $hdir_cur");
}

##tools
my $dir_bin            = $config_data->{dir_bin};
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";

my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";
my $bin_amprand        = "$dir_bin/wavAmplify_random"; ##/work/asrtrans/qrwang2/bins/2.fea_fa/wavAmplify_random

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_amprand wav out 0.3 0.05"),
::CmdToLocal("$bin_selecttail out mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_renametail out wav"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_amprand,
$bin_renametail,
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

$cmd =  "hadoop jar $bin_stream "
."-Dmapreduce.map.java.opts=\"-Xmx36000m\" "
."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
#."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=3000 "
."-Dmapreduce.reduce.memory.mb=3000 "
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
