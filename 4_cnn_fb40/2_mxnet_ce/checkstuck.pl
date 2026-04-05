#!/usr/bin/perl
use strict;
use warnings;
use List::Util 'shuffle';
$| = 1;

my ($log, $model, $maxwaittime, $outdir) = ($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3]);

my $maxwaittimeStart = 2 * 60;

my $outlog  = "$outdir/checkstuck.log";

open(OUT, ">$outlog") || die "can't open $outlog";
my $old = select(OUT); $| = 1; select($old); #### auto flush
print OUT "startcheck $log $model\n";
mkdir "$outdir/stuck";
system("echo $log >$outdir/stuck/curcheck.txt");
my $maxstrcktime = 0;
while (!-e $model && !-e "stop")
{
    if (-e $log)
    {
        my $stucktime = fileTimeDiff($log);
        $maxstrcktime = $stucktime if ($stucktime > $maxstrcktime);
        print OUT "cur check $log stucktime:$stucktime\n";
        if ($stucktime > $maxwaittime || (-s $log == 0 && $stucktime > $maxwaittimeStart))
        {
            system("cp $log $outdir/stuck");
            print OUT "check kill $log ,stucktime:$stucktime\n";
            close OUT;
            system("cp $outlog $outdir/stuck");
            system("sh $outdir/run_kill.sh");
            # system("sh $outdir/run_kill_cmd.sh");
            exit(0);
        }
    }
    sleep(3);
}
my $times = 0;
while (1)
{
    $times += 1;
    sleep(6);
    my $acc = `grep 'Validation-accuracy' $log`;
    chomp($acc);
    if ($acc || $times >= 96)
    {
        my $pid = `ps ux | grep -v grep | grep 'run_init.sh' | awk '{print \$2}'`;
        chomp($pid);
        if ($pid ne "")
        {
            print OUT "kill pid $pid\n";
            system("cp $log $outdir/stuck");
            system("cp $outlog $outdir/stuck");
            # system("sh $outdir/run_kill.sh");
            system("sh $outdir/run_kill_cmd.sh");
        }
        exit(0);
    }
    else
    {
        print OUT "wait acc in $log\n";
    }
}
print OUT "check $log finish, maxstrcktime: $maxstrcktime s \n";
close OUT;
system("cp $outlog $outdir/stuck");

sub fileTimeDiff
{
    my $file  = $_[0];
    my $mtime = (stat $file)[9] or die "can't open $file";
    my $diff  = time() - $mtime;
    return $diff;
}

