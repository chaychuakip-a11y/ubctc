use strict;

@ARGV >= 1 || die "usage: pl split_id\n";

my (@split_id)   = @ARGV;

my $dfs          = "";#"hdfs://192.168.89.100:9040";
my $dfs_param    = $dfs eq "" ? "" : "-fs $dfs";

my $hdir         = "/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_fb72_fa_state";
my $nSplit       = 10;
my $nPart        = 100;
my $states_list  = "/work1/asrdictt/taoyu/mlg/korean/am/1_mle_mfc/final_s9k/states.list";  #### cdphone cluster: cdphone.map.list.nosp; state cluster: states.list
my $scp_filter   = "";
my $descript_fea = "fb72";
my $descript_lab = "mlf_fa_ph";
my $IsCTClab     = "";  #### "0": cdphone ce label； "1": cdphone ctc label; "": state label;

##out
my $dir_pwd      = `pwd`; chomp($dir_pwd); $dir_pwd =~ s#/work1/#/yrfs4/#;
my $dir_lib      = "$dir_pwd/lib_fb72";

#tool
my $sbin_wait_hdir                     = "/work1/asrdictt/taoyu/tools/sbin/wait_dir_hdfs.pl";
my $bin_qnfiletransfer                 = "/work1/asrdictt/taoyu/tools/bin/qnfiletransfer_fast_cdph";
$bin_qnfiletransfer                    = "/work1/asrdictt/taoyu/tools/bin/qnfiletransfer_1" if($IsCTClab eq "");
my $bin_qnnorm                         = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_stat                           = "/work1/asrdictt/taoyu/tools/bin/stat_state_count_with_pfile";
my $bin_pak_low_frame_rate             = "/work1/asrdictt/taoyu/bin/pak_low_frame_rate";
my $bin_pak_low_frame_rate_labelexpand = "/work1/asrdictt/taoyu/bin/pak_low_frame_rate_labelexpand";
my $cmd;

my $partPerSplit = $nPart / $nSplit;
$nPart % $nSplit == 0 || die "Error: not support";
foreach my $split_id(@split_id)
{
	$split_id        < $nSplit || die "Error: not support";
}

$cmd = "perl $sbin_wait_hdir $hdir/_SUCCESS $dfs";
system($cmd);

foreach my $split_id(@split_id)
{
	# $split_id        = "" if($nSplit == 1);
	# my @part         = $nSplit == 1 ? ("?"x5) : map {sprintf("%05d", $_)} ($split_id*$partPerSplit..($split_id+1)*$partPerSplit-1);
	my @part         = $nSplit == 1 ? ("?"x5) : map {sprintf("%05d", $_)} (map {$_*$nSplit+$split_id} (0..$partPerSplit-1));
	my @hdir_src     = map {$_ = "$hdir/*part-$_"} @part;
	my $hdir_src     = join(" ", @hdir_src);

	my $file_norm    = "$dir_lib/fea.norm$split_id";
	my $pfile_fea    = "$dir_lib/fea.pfile$split_id";
	my $pfile_lab    = "$dir_lib/lab.pfile$split_id";
	my $scp_lab      = "$dir_lib/lab.scp$split_id";
	my $trans_log    = "$dir_lib/trans.log$split_id";

	system("mkdir -p $dir_lib") if(!-e $dir_lib);

	if(!-e "$pfile_fea.done.finish")
	{
		system("touch $pfile_fea.done.start");
		$cmd = "/home3/hadoop/hadoop2/bin/hdfs dfs $dfs_param -cat $hdir_src | $bin_qnfiletransfer - $descript_fea $pfile_fea $descript_lab $pfile_lab $states_list $scp_lab $IsCTClab $scp_filter >$trans_log 2>&1";
		# $cmd = "/home3/hadoop/hadoop2/bin/hdfs dfs $dfs_param -cat $hdir_src | $bin_pak_low_frame_rate 4 $descript_fea | $bin_qnfiletransfer - $descript_fea $pfile_fea $descript_lab $pfile_lab $states_list $scp_lab $scp_filter >$trans_log 2>&1";
		# $cmd = "/home3/hadoop/hadoop2/bin/hdfs dfs $dfs_param -cat $hdir_src | $bin_pak_low_frame_rate_labelexpand 4 $descript_fea $descript_lab | $bin_qnfiletransfer - $descript_fea $pfile_fea $descript_lab $pfile_lab $states_list $scp_lab $scp_filter >$trans_log 2>&1";
		print $cmd."\n";
		system($cmd);
		system("touch $pfile_fea.done.finish");
	}

	if(!-e "$file_norm.done")
	{
		!system("source /home3/asrdictt/taoyu/bashrc_6.5; $bin_qnnorm norm_ftrfile=$pfile_fea output_normfile=$file_norm") || die "qnnorm failed: $file_norm.\n";

		my $nStates = `wc -l $states_list | cut -d " " -f 1`;chomp($nStates);
		my $states_count = "$dir_lib/states.count.txt.tmp$split_id";
		system("$bin_stat $pfile_lab $nStates $states_count");

		system("touch $file_norm.done");
	}
}
