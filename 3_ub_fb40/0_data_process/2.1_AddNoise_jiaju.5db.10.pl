use strict;

my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;

my $jobname0     = "AddNoise_jiaju";
my $jobqueue          = $config_data->{jobqueue} || "nlp";
my $num_reduce   = 10;
my $in_blocksize = 512*1024*1024;
my $block_size   = 64*1024*1024;
my $replication  = 2;


my @hdir_src; if (@ARGV > 0) {
    @hdir_src = ($ARGV[0]);
} else {
    @hdir_src = (
                   "$config_data->{hdfs_out_root}/wav_dnnfa/*-part-*7",
                   );
}
my $hdir_out     = "$config_data->{hdfs_out_root}/wav_addnoise_jiaju_0.1.wav_dnnfa";
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

my $noisedata    = "hdfs://mycluster/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_jiaju15s_mix_all.pak/iflytek-20231116-part-00000";#IN
#my $scp_wav      = "labscp/lab.scp";  #IN
my $mlf_seed     = "out/seed.mlf"; #IN
system("$config_data->{dir_sbin}/wait_file.pl $mlf_seed.done");

my @snr          = (5);#SET
my @ratio        = (1);#SET
my $output_type  = 2;#SET
my $nSplit       = scalar(@snr);

my @splits       = (0..$#snr);
if(@ARGV >= 1)
{
	@splits = @ARGV;
	print "split: @splits\n";
}

my $bin_stream         = "$config_data->{dir_bin}/streamingAC-2.5.0.jar";
my $bin_addnoise     = "$config_data->{dir_bin}/AddNoise";
my $bin_easytraining = "$config_data->{dir_bin}/easytraining";
my $bin_selectrecord = "$config_data->{dir_bin}/selectrecord";
my $bin_selecttail   = "$config_data->{dir_bin}/selecttail";
my $bin_addtail      = "$config_data->{dir_bin}/addtail";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

#if(!-e "$scp_wav.done.split$nSplit")
#{
#	!system("$bin_easytraining -SplitScript $scp_wav $nSplit") || die "error to split file: $scp_wav, maybe the file does not exist\n";
#	system("touch $scp_wav.done.split$nSplit");
#}

@snr == @ratio || die "Error: count mismatch\n";

foreach my $i(0..$#snr)
{
	my $j = $i + 1;
	my $jobname      = $jobname0."_snr$snr[$i]";
	my $hdir_out_cur = $hdir_out;#$hdir_out."/snr$snr[$i]"
	my $snr_cur      = $snr[$i];
	my $seed_cur     = 0;
	my $ratio_cur    = $ratio[$i];
	#my $scp_wav_cur  = $scp_wav.".$j";

	@cmd_map = (
	#::CmdToLocal("$bin_selectrecord $scp_wav_cur"),
	::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
	::CmdToLocal("$bin_addtail $mlf_seed randseed"),
	::CmdToLocal("$bin_addnoise -n $noisedata -u -d -m snr_8khz -r $seed_cur -s $snr_cur -multiple $ratio_cur -output_type $output_type"),
	#::CmdToLocal("$bin_randname"),
	);

	@cmd_red = (
	#::CmdToLocal("$bin_randnamered"),
	);

	@files   = (
	$bin_selectrecord,
	$bin_selecttail,
	$bin_addtail,
	#$scp_wav_cur,
	$mlf_seed,
	$bin_addnoise,
	$noisedata,
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
	."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
	#."-Dmapreduce.map.cpu.vcores=$thread_nums "
	."-Dmapreduce.map.failures.maxpercent=20 "
	."-Dmapreduce.job.queuename=$jobqueue "
	."-Dmapreduce.job.name=$jobname "
	."-Dmapreduce.job.reduces=$num_reduce "
	."-Dmapreduce.map.memory.mb=4000 "
	."-Dmapreduce.reduce.memory.mb=1500 "
	."-Ddc.input.block.size=$in_blocksize "
	."-Ddfs.blocksize=$block_size "
	."-Ddfs.replication=$replication "
	."-files \"$files\" "
	."-input $hdir_src "
	."-output $hdir_out_cur "
	;

	$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
	$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

	::RemoveHadoopDirIfExist($hdir_out_cur);
	::PR($cmd);
	::SuccessOrDie("$hdir_out_cur");
}

#foreach my $i(1..$nSplit)
#{
#	system("rm -rf $scp_wav.$i");
#}
