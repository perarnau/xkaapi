dnl KAT_CHECK_HEADERS(subdir, testsuite, extra_keywords) 
m4_define([KAT_CHECK_HEADERS],[
  AT_SETUP([Checking whether header files compile])
  AT_KEYWORDS([$1 $2 headers $3])
  AT_SKIP_IF([AS_CASE([" $KAT_TESTSUITES "],[*" $2 "*],[false],[:])])
  AT_CHECK([make -C ${abs_top_builddir}/tests/$1 check_header ], 0, [ignore])
  AT_CHECK([make -C ${abs_top_builddir}/tests/$1 check_header_strict ], 0, [ignore])
  AT_CLEANUP
])
