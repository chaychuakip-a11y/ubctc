#!/usr/bin/perl
use strict;
use warnings;

my $old = '    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;';

my $new = '    if (defined $config_data->{dir_sbin} && $config_data->{dir_sbin} ne \'\') {
        unshift @INC, $config_data->{dir_sbin};
    } else {
        die "Error: dir_sbin not set in config.json (needed to locate share_hadoop.pm)\n";
    }
    eval { require share_hadoop; share_hadoop->import(); };
    if ($@) {
        die "Error: cannot load share_hadoop from dir_sbin=$config_data->{dir_sbin}\n"
          . "  Check that share_hadoop.pm exists at that path.\n"
          . "  Current \@INC: " . join(", ", @INC) . "\n";
    }
}';

my @files = glob("*.pl");
my $count = 0;
for my $file (@files) {
    next if $file =~ /^(utils|fix_inc|replace_headers|replace_paths|_fix_share_hadoop)\.pl$/;
    open(my $fh, "<", $file) or die "can't open $file: $!";
    my $content = do { local $/; <$fh> };
    close($fh);

    # normalize to LF for matching, remember original line ending style
    my $has_crlf = ($content =~ /\r\n/);
    (my $normalized = $content) =~ s/\r\n/\n/g;

    if (index($normalized, $old) >= 0) {
        $normalized =~ s/\Q$old\E/$new/;
        $normalized =~ s/\r\n/\n/g; # ensure no mixed endings
        $content = $has_crlf ? do { (my $c = $normalized) =~ s/\n/\r\n/g; $c } : $normalized;
        open(my $out, ">", $file) or die "can't write $file: $!";
        print $out $content;
        close($out);
        print "fixed: $file\n";
        $count++;
    }
}
print "total: $count files fixed\n";
