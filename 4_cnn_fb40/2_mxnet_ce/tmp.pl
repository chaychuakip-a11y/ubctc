#!/usr/bin/perl -w
use strict;
use warnings;

my ($file)          = @ARGV;

my $curCV = ::GetCV($file);
print($curCV);

sub GetCV
{
    my ($log) = @_;
    my $curCV = 0;
    my $count = 0;
    open(IN, $log) || die;
    while (<IN>)
    {
        if (/Validation-accuracy=(\S+)/)
        {
            $count++;
            $curCV += $1;
        }

    }
    close IN;
    if ($count > 0)
    {
        $curCV /= $count;
        $curCV = int($curCV * 10000 + 0.5) / 10000;
    }
    return $curCV;
}
