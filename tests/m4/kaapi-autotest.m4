
m4_include([m4/atenv.m4])
m4_include([check-headers/check-headers.m4])

dnl Try to avoid typo in macro names
m4_pattern_forbid([^_?KAT_])
dnl But some used shell variables must be allowed
m4_pattern_allow([^(KAT_TESTSUITES|KAT_EX_CURRENT_LIST)$])

dnl **********************************************
dnl Hook system
dnl **********************************************

dnl Cleanup macro hooks (called internally later)
dnl _KAT_HOOK_INIT_PREFIX(prefix, blocklist)
m4_define([_KAT_HOOK_INIT_PREFIX],[
  m4_foreach([_KAT_HOOK_BLOCK], [$2], [
    m4_foreach([_KAT_HOOK_WHEN], [[BEFORE],[AFTER]], [
      m4_foreach([_KAT_HOOK_TYPE], [[_KAT_HOOK_CODE],[_KAT_HOOK_MACRO]], [
        m4_define(_KAT_HOOK_TYPE[$1_]_KAT_HOOK_BLOCK[_]_KAT_HOOK_WHEN,[])
      ])
    ])
  ])
])
dnl _KAT_HOOK_INIT(block1, block2, ...)
m4_define([_KAT_HOOK_INIT],[
  m4_define([_KAT_HOOK_LIST_BLOCK],[$@])
  _KAT_HOOK_INIT_PREFIX([],[_KAT_HOOK_LIST_BLOCK])
  _KAT_HOOK_INIT_PREFIX([ONCE],[_KAT_HOOK_LIST_BLOCK])
])
dnl Run hook associated to a block
dnl _KAT_HOOK_RUN(block, when)
m4_define([_KAT_HOOK_RUN],[
  _KAT_HOOK_CODEONCE_$1_$2
  _KAT_HOOK_MACROONCE_$1_$2(m4_shift(m4_shift($@)))
  m4_define([_KAT_HOOK_CODEONCE_$1_$2],[])
  m4_define([_KAT_HOOK_MACROONCE_$1_$2],[])
  _KAT_HOOK_CODE_$1_$2
  _KAT_HOOK_MACRO_$1_$2(m4_shift(m4_shift($@)))
])

dnl User macros
dnl ===========

dnl Add code before or after a block
dnl KAT_HOOK(block, when, code)
m4_define([KAT_HOOK],[
  m4_define([_KAT_HOOK_CODE_$1_$2],[$3])
])
dnl Add a macro to be excuted with the same parameter as the targer block
dnl KAT_HOOK_MACRO(block, when, macro-name)
m4_define([KAT_HOOK_MACRO],[
  m4_define([_KAT_HOOK_MACRO_$1_$2],[$3($]][[@)])
])
dnl Add code before or after a block, only for one occurence
dnl KAT_HOOK(block, when, code)
m4_define([KAT_HOOK_ONCE],[
  m4_define([_KAT_HOOK_CODEONCE_$1_$2],[$3])
])
dnl Idem for macro
dnl KAT_HOOK_MACRO(block, when, macro-name)
m4_define([KAT_HOOK_MACRO_ONCE],[
  m4_define([_KAT_HOOK_MACROONCE_$1_$2],[$3($]][[@)])
])

dnl **********************************************
dnl Chose whether we are doing tests after or before the installation
dnl **********************************************

m4_define([KAT_INSTALLED],[m4_if(_KAT_INSTALLED,true,[$1],[$2])])
m4_define([KAT_BUILDING],[KAT_INSTALLED([$2],[$1])])

dnl **********************************************
dnl Convenience macros to do tests for our testsuite
dnl **********************************************

dnl KAT_BANNER(name, testsuite, keywords, subdir, all-target, options-for-all-target)
m4_define([KAT_BANNER],[
  AT_BANNER([$1])
  m4_define([_KAT_TESTSUITE],[$2])
  m4_define([_KAT_KEYWORDS],[$2 $3])
  m4_define([_KAT_SUBDIR],[$4])
  m4_define([_KAT_WRAPPER],[$6])
  _KAT_HOOK_INIT([SETUP],[CHECK_BUILD],[CHECK_RUN])

  m4_if($5,[],[],[
    KAT_SETUP([clean compilation into _KAT_SUBDIR], [clean])
    KAT_CHECK_BUILD([clean])
    AT_CLEANUP
  
    KAT_SETUP([precompile programs into _KAT_SUBDIR], [build])
    KAT_CHECK_BUILD([$5],[-k $6],[ignore])
    AT_CLEANUP
  ])
])
dnl KAT_SETUP(name, extra_keywords) 
m4_define([KAT_SETUP],[
  AT_SETUP([$1])
  _KAT_HOOK_RUN([SETUP],[BEFORE],$@)
  AT_KEYWORDS([_KAT_KEYWORDS $2])
  m4_if(_KAT_TESTSUITE,[],[],[
    AT_SKIP_IF([AS_CASE([" $KAT_TESTSUITES "],[*" _KAT_TESTSUITE "*],[false],[:])])
  ])
  _KAT_HOOK_RUN([SETUP],[AFTER],$@)
])
dnl KAT_CHECK_BUILD(target, options, exit-code)
m4_define([KAT_CHECK_BUILD],[
  _KAT_HOOK_RUN([CHECK_BUILD],[BEFORE],$@)
  AT_CHECK([${MAKE:-make} -C ${abs_top_builddir}/_KAT_SUBDIR $2 $1], 
    m4_if([$3],[],[0],[$3]), [ignore], [ignore])
  _KAT_HOOK_RUN([CHECK_BUILD],[AFTER],$@)
])
dnl KAT_CHECK_RUN(name, arguments, command-to-parse-stdout, stdout, stderr)
m4_define([KAT_CHECK_RUN],[
  _KAT_HOOK_RUN([CHECK_RUN],[BEFORE],$@)
  AT_CHECK([_KAT_WRAPPER $abs_top_builddir/_KAT_SUBDIR/$1 $2], 0,[m4_if([$3],[],[$4],[stdout])],[$5])
  m4_if([$3],[],[],[
    AT_CHECK([$3], 0, [$4], [])
  ])
  _KAT_HOOK_RUN([CHECK_RUN],[AFTER],$@)
])
dnl KAT_TEST_PROG(prog-name, test-name, keywords, , arguments, command-to-parse-stdout, stdout, stderr)
m4_define([KAT_TEST_PROG],[
  KAT_SETUP([$2],[$3 $1])
  KAT_CHECK_BUILD([$1])
  KAT_CHECK_RUN([$1],[$5],[$6],[$7],[$8])
  AT_CLEANUP
])
dnl KAT_TEST_BUILD_FAILED(prog-name, test-name, keywords)
m4_define([KAT_TEST_BUILD_FAILED],[
  KAT_SETUP([$2],[$3 $1])
  KAT_CHECK_BUILD([$1], [], [1])
  AT_CLEANUP
])

