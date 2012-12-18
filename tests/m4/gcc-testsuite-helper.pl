#!/usr/bin/perl

# tools to 'autogenerate' autotest entries from a complete DejaGNU runtest
# this file must not be distributed (nor used again without very good reasons)

my $fail=0;
my $failtxt='';
my $timeout=0;
my $file='';

sub filereset() {
  $fail=0;
  $failtxt='';
  $timeout=0;
}

sub filedump() {
  if (!$fail) {
    print "KAT_GLT_TEST([$file])\n";
  } else {
    print "KAT_GLT_TEST_FAIL([$file],[";
    if ($timeout) {
      print "timeout";
    }
    print "],[dnl\n", $failtxt,"])\n";
  }
  filereset();
}

while(<>) {
  chomp;
  next if (! /^[A-Z]*: /);
  my $start = $_;
  $start =~ s/:.*//;
  if ($start eq 'WARNING') {
    $fail=1;
    $timeout=1;
    $failtxt .= $_."\n"
  } else {
    my $cur_file = $_;
    $cur_file =~ s/^[A-Z]*: ([^ ]+) .*$/$1/;
    if ($cur_file ne $file) {
      if ($file ne '') {
        filedump();
      }
      $file = $cur_file;
    }
    if ($start ne 'PASS') {
      $fail=1;
      $failtxt .= $_."\n"
    }
  }
}
filedump()
