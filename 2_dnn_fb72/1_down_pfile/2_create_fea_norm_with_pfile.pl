#!/usr/bin/perl -w
use strict;

my $dir_pwd      = `pwd`; chomp($dir_pwd); $dir_pwd =~ s#/work1/#/yrfs4/#;
my $dir_lib      = "$dir_pwd/lib_fb72";
my $dir_lib0     = "./lib_fb72"; mkdir($dir_lib0) if(!-e $dir_lib0);
my $nSplit       = 10;
my $file_norm    = "$dir_lib0/fea.norm";
my @pfile_fea    = $nSplit == 1 ? ("$dir_lib/fea.pfile0") : map {$_ = "$dir_lib/fea.pfile$_"} (0..$nSplit-1);
my @pfile_lab    = $nSplit == 1 ? ("$dir_lib/lab.pfile0") : map {$_ = "$dir_lib/lab.pfile$_"} (0..$nSplit-1);
my @file_norm    = $nSplit == 1 ? ("$dir_lib/fea.norm0" ) : map {$_ = "$dir_lib/fea.norm$_" } (0..$nSplit-1);

my $bin_qnnorm          = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_pfile_info      = "/work1/asrdictt/taoyu/tools/QN/bin/pfile_info";
my $sbin_wait_file      = "/work1/asrdictt/taoyu/sbin/wait_file.pl";

my $pfile_fea_cur;
my $file_norm_cur;
my $j;
my @nFrame;

if(!-e "$file_norm.done")
{
	foreach $j(0..$nSplit-1)
	{
		$pfile_fea_cur = $pfile_fea[$j];
		$file_norm_cur = $file_norm[$j];
		if(!-e "$file_norm_cur.done")
		{
			#!system("$bin_qnnorm norm_ftrfile=$pfile_fea_cur output_normfile=$file_norm_cur") || die "qnnorm failed: $file_norm_cur.\n";
			#system("touch $file_norm_cur.done");
			system("$sbin_wait_file $file_norm_cur.done");
		}
		if(!defined $nFrame[$j])
		{
			$nFrame[$j] = `$bin_pfile_info $pfile_fea_cur | tail -n 1 | cut -d " " -f 3`; chomp $nFrame[$j];
		}
	}
	if($nSplit >1)
	{
		MergeNormFile($file_norm, \@file_norm, \@nFrame);
	}
	else
	{
		system("cp -p $file_norm[0] $file_norm") if($file_norm ne "$file_norm[0]");
	}
	system("touch $file_norm.done");
}

my $dim = `wc -l $file_norm | cut -d ' ' -f 1`; chomp($dim);
$dim = ($dim-2)/2;

if($dim == 160)
{
	system("/work1/asrdictt/taoyu/sbin/fea_norm.lfr2normal.pl $file_norm 4");
}

sub MergeNormFile
{
	@_ == 3 || die "Usage: MergeNormFile(file_norm, ref_file_norm, ref_nFrame)\n";
	my ($file_norm, $ref_file_norm, $ref_nFrame) = @_;
	my @file_norm = @{$ref_file_norm};
	my @nFrame    = @{$ref_nFrame};

	@file_norm == @nFrame || die "Error: count mismatch: file_norm vs nFrame\n";

	my @mean_all;
	my @var_all;
	my $i;
	my $j;

	foreach $j(0..$#file_norm)
	{
		my $nDim;
		my @mean;
		my @var;
		my $file_norm_cur = $file_norm[$j];
		open(IN, $file_norm_cur) || die "Error: cannot read file: $file_norm_cur\n";
		$_ = <IN>;chomp;
		/^vec\s+(\d+)/ || die "Informal line: $_";
		$nDim = $1;
		while(<IN>)
		{
			chomp;
			push(@mean, $_);
			$nDim--;
			last if($nDim == 0);
		}
		$_ = <IN>;chomp;
		/^vec\s+(\d+)/ || die "Informal line: $_";
		$nDim = $1;
		while(<IN>)
		{
			chomp;
			push(@var, $_);
			$nDim--;
			last if($nDim == 0);
		}
		close IN;
		push(@mean_all, [@mean]);
		push(@var_all, [@var]);

		@mean == @var || die;
		@mean == @{$mean_all[0]} || die;
	}

	my @mean;
	my @var;
	my $nFrameTotal = 0;
	foreach $j(0..$#file_norm)
	{
		$nFrameTotal += $nFrame[$j];
	}
	foreach $i(0..$#{$mean_all[0]})
	{
		foreach $j(0..$#file_norm)
		{
			$mean[$i] += $nFrame[$j]/$nFrameTotal*${$mean_all[$j]}[$i];
			$var[$i] += $nFrame[$j]/$nFrameTotal*(1.0/${$var_all[$j]}[$i]/${$var_all[$j]}[$i] + ${$mean_all[$j]}[$i]*${$mean_all[$j]}[$i]);
		}
		$var[$i] -= $mean[$i]*$mean[$i];
		$var[$i] = 1/sqrt($var[$i]);
	}

	open(OUT, ">", $file_norm) || die "Error: cannot write file: $file_norm\n";
	print OUT "vec ".scalar(@mean)."\n";
	foreach (@mean)
	{
		printf OUT "%.6e\n", $_;
	}
	print OUT "vec ".scalar(@var)."\n";
	foreach (@var)
	{
		printf OUT "%f\n", $_;
	}
	close OUT;
}

1;
