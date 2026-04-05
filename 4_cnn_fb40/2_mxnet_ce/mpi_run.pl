#!/usr/bin/perl
use strict;
use warnings;
$| = 1; #### flush STDOUT

my $ngpu = $ARGV[0];

system("rm -rf stop");
my $count = 0;
while (!-e "stop")
{
    # system("perl train_init.pl -1");
    system("perl train.pl $ngpu >train.$count.log 2>&1");
    sleep(60);
    $count++;
    print "MPI run died ,restart $count\n";
}
