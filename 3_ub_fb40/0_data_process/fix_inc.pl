use strict;
use warnings;
use File::Basename;

my @files = glob("*.pl");
foreach my $file (@files) {
    next if $file eq 'utils.pl' || $file eq 'fix_inc.pl';
    
    my $content;
    {
        local $/;
        open(my $fh, "<", $file) or die "can't open $file: $!";
        $content = <$fh>;
        close($fh);
    }

    my $old_block = qr/my \$config_data;\s*BEGIN \{\s*require "\.\/utils\.pl";\s*\$config_data = load_config\(\);\s*\}\s*use lib \$config_data->\{dir_sbin\};\s*use share_hadoop;/s;

    my $new_block = 'my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        require lib;
        lib->import($config_data->{dir_sbin});
    }
}
use share_hadoop;';

    if ($content =~ s/$old_block/$new_block/) {
        open(my $fh_out, ">", $file) or die "can't write to $file: $!";
        print $fh_out $content;
        close($fh_out);
        print "Updated $file\n";
    }
}
