use strict;
use warnings;
use File::Basename;

my @files = (
    '1_dnnfa.pl',
    '2.0_GenSeedMlf.pl',
    '2.1_AddNoise_car_byd.5db.20.pl',
    '2.1_AddNoise_car_dz.5db.20.pl',
    '2.1_AddNoise_duodian.5db.10.pl',
    '2.1_AddNoise_gs.5db.20.pl',
    '2.1_AddNoise_jiaju.5db.10.pl',
    '2.1_AddNoise_music_a.5db.2.pl',
    '2.1_AddNoise_music_b.5db.2.pl',
    '2.1_AddNoise_music_c.5db.2.pl',
    '2.1_AddNoise_music_onenoise.5db.2.pl',
    '2.1_AddNoise_music_tv.5db.2.pl',
    '2.1_AddNoise_pingwen.5db.10.pl',
    '3.0_speedup1.2.pl',
    '3.1_dnnfa.pl',
    '4.0_amp.pl',
    '6.0_LsaDenoise.pl',
    '6.1_dnnfa.pl',
    '7.0_MaeDenoiseClose.pl',
    '7.1_dnnfa.pl',
    '8.0_MaeDenoiseOpen.pl',
    '8.1_dnnfa.pl',
    '9_fea_merge.pl'
);

my $new_header = 'my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;
';

foreach my $file (@files) {
    if (!-e $file) {
        warn "File $file not found, skipping.\n";
        next;
    }

    my $content;
    {
        local $/;
        open(my $fh, "<", $file) or die "can't open $file: $!";
        $content = <$fh>;
        close($fh);
    }

    # Replacement 1: Replace @INC with @INC
    # The user said "Replace any occurrences of "@INC" back to "@INC""
    # Although grep didn't find it, we'll do it anyway to be safe.
    $content =~ s/\@fix_inc\.pl/\@INC/g;

    # Replacement 2: Replace the beginning of the file
    # We need to find the old header and replace it.
    # The old header typically starts with (optional empty lines/shebang) + use strict;
    # and ends with use share_hadoop; or the end of the BEGIN block.
    
    # Let's try to match the existing block pattern.
    # Style 1 (most files):
    # use strict;
    # 
    # my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;
    
    # Style 2 (2.0_GenSeedMlf.pl):
    # use strict;
    # 
    # my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;' or the end of the BEGIN block that has load_config.
    
    # Actually, a more surgical approach:
    # Look for the block that starts with 'my $config_data;' and ends with 'use share_hadoop;' or '}' of BEGIN.
    
    # But wait, some files have "use strict;" above "my $config_data;".
    # The user's new structure replaces the BEGINNING of the file.
    
    # Let's try this:
    if ($content =~ s/^\s*(?:#!.*?\n)?\s*(?:use strict;\s*)?my \$config_data;.*?BEGIN\s*\{.*?load_config\(\);.*?\}(?:\s*use share_hadoop;)?/$new_header/s) {
        open(my $fh_out, ">", $file) or die "can't write to $file: $!";
        print $fh_out $content;
        close($fh_out);
        print "Updated $file\n";
    } else {
        warn "Could not find header pattern in $file\n";
    }
}
