#!/usr/bin/perl -w
use strict;

my $dir_pwd      = `pwd`; chomp($dir_pwd); $dir_pwd =~ s#/work1/#/yrfs4/#;
my $dir_lib      = "$dir_pwd/lib_fb40";
my $dir_lib0     = "./lib_fb40"; mkdir($dir_lib0) if(!-e $dir_lib0);
my $nSplit       = 10;
my $states_list  = "/work1/asrdictt/taoyu/mlg/korean/am/1_mle_mfc/final_s9k_for_ctc/states.list.nosp"; #### cdphone cluster: states.list.nosp; state cluster: states.list
my $states_count = "$dir_lib0/states.count.txt";

my $bin_stat     = "/work1/asrdictt/taoyu/tools/bin/stat_state_count_with_pfile";

-e $states_list || die "Error: no exists file: $states_list\n";

my $nStates = `wc -l $states_list | cut -d " " -f 1`;chomp($nStates);

print "number of states: $nStates\n";

foreach my $i(0..$nSplit-1)
{
	#$i = "" if($nSplit == 1);
	my $states_count_cur = "$dir_lib/states.count.txt.tmp$i";
	my $pfile_cur = "$dir_lib/lab.pfile$i";

	system("$bin_stat $pfile_cur $nStates $states_count_cur") if(!-s $states_count_cur);
}

my @states;
my @states_count;
open(IN, $states_list) || die "Error: cannot read file: $states_list\n";
@states = <IN>; chomp(@states);
close IN;
foreach my $i(0..$#states)
{
	$states_count[$i] = 0;
}

my $total_frame = 0;
foreach my $i(0..$nSplit-1)
{
	#$i = "" if($nSplit == 1);
	my $states_count_cur = "$dir_lib/states.count.txt.tmp$i";

	open(IN, $states_count_cur) || die "Error: cannot read file: $states_count_cur\n";
	$_ = <IN>;
	while(<IN>)
	{
		chomp;
		if(/(\d+)\s+(\d+)/)
		{
			$states_count[$1] += $2;
			$total_frame += $2;
		}
		else
		{
			die "Error: informal line: $_";
		}
	}
	close IN;

	#unlink $states_count_cur;
}

open(OUT, ">", $states_count) || die "Error: cannot write file: $states_count\n";
print OUT "$nStates\t$total_frame\n";
foreach my $i(0..$#states)
{
	print OUT "$states[$i]\t$i\t$states_count[$i]\n";
}
close OUT;

# system("/work1/asrdictt/taoyu/sbin/prior_process.low.pl $dir_lib0/states.count.txt $dir_lib0/states.count.low100.txt 100");
# system("/work1/asrdictt/taoyu/sbin/prior_process.add_vad.pl $dir_lib0/states.count.low100.txt $dir_lib0/states.count.low100.vad.txt");
system("/work1/asrdictt/taoyu/sbin/prior_process.ctc.pl $dir_lib0/states.count.txt $dir_lib0/states.count.ctc-2.txt -2");
system("/work1/asrdictt/taoyu/sbin/prior_process.ctc.pl $dir_lib0/states.count.txt $dir_lib0/states.count.ctc-1.txt -1");

1;
