dnl KAT_TEST_HEADERS(extra_keywords) 
m4_define([KAT_TEST_HEADERS],[
  KAT_SETUP([Checking whether header files compile], [$1])
  KAT_CHECK_BUILD([check_header])
  KAT_CHECK_BUILD([check_header_strict])
  AT_CLEANUP
])
