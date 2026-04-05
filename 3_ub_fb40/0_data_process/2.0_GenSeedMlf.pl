use strict;
use List::Util 'shuffle';

my $hdir     = "/workdir/asrdictt/tasrdictt/taoyu/mlg/korean/17kh_wav_dnnfa";
my $nSplit   = 10;
my $nPart    = 100;
my $dir_out  = "out";

my @hdir_src = ($hdir);
foreach my $hdir_cur (@hdir_src)
{
	if($hdir_cur =~ /\*/ && $hdir_cur =~ /part/)
	{
		$hdir_cur =~ s#/[^/]+$##;
	}
	$hdir_cur .= '/_SUCCESS';
	system("/work1/asrdictt/taoyu/sbin/wait_dir_hdfs.pl $hdir_cur");
}

my $partPerSplit = $nPart / $nSplit;
$nPart % $nSplit == 0 || die "Error: not support";

mkdir $dir_out if !-e $dir_out;
foreach my $split_id(0..$nSplit-1)
{
	my @part         = $nSplit == 1 ? ("?"x5) : map {sprintf("%05d", $_)} ($split_id*$partPerSplit..($split_id+1)*$partPerSplit-1);
	my $hdir_cur     = join(" ", map {$_ = "$hdir/*part-$_"} @part);
	my $cmd          = "hdfs dfs -cat $hdir_cur | /work1/asrdictt/taoyu/bin/fea_lab_lat_unpack_1 - $dir_out/split$split_id; touch $dir_out/split$split_id/done";
	system("nohup sh -c \"$cmd\" >$dir_out/split$split_id.log 2>&1 &")
}

foreach my $split_id(0..$nSplit-1)
{
	system("/work1/asrdictt/taoyu/sbin/wait_file.pl $dir_out/split$split_id/done");
}

my $files = join(" ", map {$_ = "$dir_out/split$_/out.record.scp"} (0..$nSplit-1));

system("cat $files >$dir_out/out.record.scp");

my $scp_in  = "$dir_out/out.record.scp";
my $scp_out = "$dir_out/seed.mlf";
my $seed    = 100;

my (@list, @idx, $num);

srand($seed) if(defined($seed));

open(IN, "$scp_in") || die $!;
@list = <IN>;
map chomp, @list;
@idx = (0..$#list);
close(IN);

@idx = shuffle(@idx);

open(OUT, ">", $scp_out) || die $!;
my $i = 0;
foreach my $list (@list)
{
	$list =~ s/.*\///;
	$list =~ s/\.lab//;
	#my($id, $file) = split(/=/, $list);
	#my $id = join("-", $list, $idx[$i]);
	print OUT "\"*\/$list.lab\"\n";
	print OUT $idx[$i]."\n";
	print OUT "."."\n";
	$i ++;
}
close OUT;

system("touch $scp_out.done")
