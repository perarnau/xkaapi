#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.60])

AC_INIT([X-KAAPI],[1.0.4],[kaapi-dev-info@lists.gforge.inria.fr],[xkaapi])
dnl AC_CONFIG_AUX_DIR: not before, not after
dnl http://www.mail-archive.com/autoconf@gnu.org/msg07824.html
AC_CONFIG_AUX_DIR([build-aux])
AC_CANONICAL_TARGET
AC_CONFIG_SRCDIR([src/workstealing/kaapi_sched_idle.c])
AC_CONFIG_MACRO_DIR([acinclude])

dnl automake 1.6 is supposed NOT to work
dnl automake 1.9 is supposed to work
dnl automake 1.10 is known to work
dnl other versions have not been tested (please, report if you want
dnl we downgrade/ajust the requirement)
AM_INIT_AUTOMAKE([gnu 1.9 tar-pax subdir-objects])
dnl silent-rules option is not required as it has been introduced in automake 1.11
dnl The AMX_SILENT_RULES macro enables it if it exists


AC_CONFIG_HEADERS([config.h])

AMX_SILENT_RULES([AM_DEFAULT_VERBOSITY=0])

dnl Sanity checks on prefix : do not install kaapi in the source tree
dnl nor in the build tree
if test -d "$prefix" ; then
   prefix_pwd=`cd $prefix ; pwd`
   if test x"$prefix_pwd" = x`test -n "$srcdir" && cd "$srcdir" ; pwd` ; then
      AC_MSG_WARN([Invalid installation prefix '$prefix':])
      AC_MSG_ERROR([KAAPI cannot be installed at the root of the source directory])
   fi
   if test x"$prefix_pwd" = x`test -n "$builddir" && cd "$builddir" ; pwd` ; then
      AC_MSG_WARN([Invalid installation prefix '$prefix':])
      AC_MSG_ERROR([KAAPI cannot be installed at the root of the build directory])
   fi
fi

git_repo=yes
dnl try to find if we are in a git repo
GIT_LAST_COMMIT="$(cd "$srcdir" ; git log -n 1 --format="format:%H" HEAD 2>/dev/null)"
if test -z "$GIT_LAST_COMMIT" ; then
  git_repo=no
fi
AM_CONDITIONAL([GIT_REPO],[test $git_repo = yes])

dnl do not change order of --with and --enable options
AC_PRESERVE_HELP_ORDER

dnl ----------------------------------------------------------------
dnl Checks for programs and compilers options
dnl ----------------------------------------------------------------
PKG_PROG_PKG_CONFIG
AC_PROG_CC
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_PROG_INSTALL
LT_INIT
ACX_PROG_CCACHE_CC
AC_PROG_CXX
ACX_PROG_CCACHE_CXX
AC_PROG_F77

AC_MSG_CHECKING([whether $CXX works])
AC_LANG_PUSH([C++])
AC_COMPILE_IFELSE(
            [AC_LANG_PROGRAM([],[return 0;])],
            [have_cxx=yes],
            [have_cxx=no])
AC_LANG_POP([C++])
AC_MSG_RESULT([$have_cxx])
AM_CONDITIONAL([HAVE_CXX],[test $have_cxx = yes])

# if next line is comented out, we can create only dynamic libraries be default
# AC_DISABLE_STATIC
LT_INIT
AC_SUBST(LIBTOOL_DEPS)

AC_PROG_MKDIR_P
AX_PROG_ESED

AC_CACHE_SAVE

NESTED_FUNCTIONS_CFLAGS=""
AX_CHECK_COMPILER_FLAGS([-fnested-functions],
  [NESTED_FUNCTIONS_CFLAGS="-fnested-functions"])
AC_SUBST([NESTED_FUNCTIONS_CFLAGS])

AC_LANG_PUSH([C])
save_CC="$CC"
AC_PROG_CC_C99
STD_C99_FLAGS=""
if test x"$ac_cv_prog_cc_c99" = xno ; then
   AC_MSG_WARN([Unable to select C99 standard with the current C compiler.])
   AC_MSG_WARN([Pedantic mode is disabled and some warning and/or errors can occur])
   STD_C99_FLAGS="-D_KAAPI_DISABLE_WARNINGS"
   STD_C99=no
else
   for arg in $CC ; do
     AS_CASE([" $save_CC "],
       [*" $arg "*], [],
       [STD_C99_FLAGS="$STD_C99_FLAGS $arg"])
   done
   AX_CHECK_COMPILER_FLAGS([-pedantic],
     [ STD_C99_FLAGS="$STD_C99_FLAGS -pedantic" ])
   STD_C99=yes
fi 
AC_SUBST([STD_C99_FLAGS])
AM_CONDITIONAL([STD_C99], [test x"$STD_C99" = xyes])
CC="$save_CC"
AC_LANG_POP([C])

STD_CXX0X_FLAGS=""
STD_CXX0X=no
if test $have_cxx = yes ; then
  AC_LANG_PUSH([C++])
  AX_CHECK_COMPILER_FLAGS([-std=c++0x],
    [STD_CXX0X=yes
     AX_CHECK_COMPILER_FLAGS([-pedantic],
       [STD_CXX0X_FLAGS="-std=c++0x -pedantic"],
       [STD_CXX0X_FLAGS="-std=c++0x"])
    ],
    [
     STD_CXX0X_FLAGS="-D_KAAPI_DISABLE_WARNINGS"
     AC_MSG_WARN([Unable to select C++0X standard with the current C++ compiler.])
     AC_MSG_WARN([Pedantic mode is disabled and some warning and/or errors can occur])
    ])
  AC_LANG_POP([C++])
fi
AC_SUBST([STD_CXX0X_FLAGS])
AM_CONDITIONAL([STD_CXX0X], [test x"$STD_CXX0X" = xyes])

if test $have_cxx = yes ; then
  CXXFLAGS_save="$CXXFLAGS"
  CXXFLAGS="$CXXFLAGS $STD_CXX0X_FLAGS"
  AC_CXX_LAMBDA()
  CXXFLAGS="$CXXFLAGS_save"
fi
AM_CONDITIONAL([KAAPI_CXX_HAS_LAMBDA],[test x"$ac_cv_cxx_lambda" = x"yes"])

# TODO: better check for this feature
AM_CONDITIONAL([HAVE_VERSION_SCRIPT],[test x"$with_gnu_ld" = xyes])
if test x"$with_gnu_ld" = xyes; then
	AC_DEFINE([HAVE_VERSION_SYMBOL], [1], [Define to 1 if symbols can be versionned.])
fi

# add warning flags to the C and C++ compilers
AX_CFLAGS_WARN_ALL([AM_CFLAGS])
if test $have_cxx = yes ; then
  AX_CXXFLAGS_WARN_ALL([AM_CXXFLAGS])
fi

# check for __attribute__((visibility("..."))) support
ACX_VISIBILITY

# check for automatic macros that are used by KAAPI
AX_MACRO_DEFINED([__SIZEOF_POINTER__],[],[],[
  AC_MSG_WARN([The '__SIZEOF_POINTER__' macro is not defined by the compiler. Support for such old compilers will be removed in a future release])
])

dnl ----------------------------------------------------------------
dnl Define libs and progs created by this software
dnl ----------------------------------------------------------------
ACX_LIB_NEW_LIB([KAAPILIB], [-L\${libdir} -lkaapi], [kaapi])
ACX_LIB_ADD_PUBLIC_FLAGS([KAAPILIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([KAAPIXXLIB], [-L\${libdir} -lkaapi++], [kaapi++])
ACX_LIB_ADD_PUBLIC_FLAGS([KAAPIXXLIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([KANETLIB], [-L\${libdir} -lkanet], [kanet])
ACX_LIB_ADD_PUBLIC_FLAGS([KANETLIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([KASTLLIB], [-L\${libdir}], [kastl])
ACX_LIB_ADD_PUBLIC_FLAGS([KASTLLIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([ATHALIB], [-L\${libdir} -latha], [atha])
ACX_LIB_ADD_PUBLIC_FLAGS([ATHALIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([KAAPICLIB], [-L\${libdir} -lkaapic], [kaapic])
ACX_LIB_ADD_PUBLIC_FLAGS([KAAPICLIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([KAAPIFLIB], [-L\${libdir} -lkaapif], [kaapif])
ACX_LIB_ADD_PUBLIC_FLAGS([KAAPIFLIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([LIBGOMP], [-L\${libdir} -lgomp], [])

ACX_LIB_NEW_PRG([KAAPIPROGS])
ACX_LIB_NEW_PRG([KAAPIXXPROGS])
ACX_LIB_NEW_PRG([KAAPICPROGS])
ACX_LIB_NEW_PRG([KAAPIFPROGS])
ACX_LIB_NEW_PRG([KANETPROGS])
ACX_LIB_NEW_PRG([KASTLPROGS])
ACX_LIB_NEW_PRG([ATHAPROGS])

default_experimental=no

# check for automatic macros that are used by KAAPI
AX_MACRO_DEFINED([__SIZEOF_POINTER__],[],[],[
  AC_MSG_ERROR([The '__SIZEOF_POINTER__' macro must be defined by the compiler. Try a recent version of gcc for example])
])

dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl Selection of APIs
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------

dnl -----------------------------------------------------------------------
dnl The "--enable-api-kaapic"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-kaapic], [compile C API],
  [C API compiled], [yes],
  [],
  [],
  [KAAPI_API_KAAPIC])

dnl -----------------------------------------------------------------------
dnl The "--enable-api-kaapif"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-kaapif], [compile fortran API],
  [Fortran API compiled], [check],
  [],
  [],
  [KAAPI_API_KAAPIF],
  [
    if test x"$F77" = x; then
      AC_MSG_WARN([a Fortran compiler is required for Fortran API])
      FEATURE_DISABLE
    else
dnl      ACX_LIB_CHECK_PLAIN([FORTRAN],[],[
dnl        AC_F77_LIBRARY_LDFLAGS
dnl        ACX_LIB_FOUND([Libs="$F77LIBS"])
        FEATURE_ENABLE
dnl      ],[
dnl        FEATURE_DISABLE
dnl      ])
    fi
  ])
dnl ACX_LIB_LINK([KAAPIFLIB], [FORTRAN])

dnl KAAPI_FEATURE([api-kaapif], [compile fortran API],
dnl   [fortran API compiled], [$default_experimental],
dnl   [ ],
dnl   [ 
dnl     ACX_LIB_NOLIB([FORTRAN])
dnl   ],
dnl   [KAAPI_API_KAAPIF],
dnl   [
dnl     if test x"$F77" = x; then
dnl       FEATURE_DISABLE
dnl     else
dnl       ACX_LIB_CHECK_PLAIN([FORTRAN],[],[
dnl         AC_F77_LIBRARY_LDFLAGS
dnl         ACX_LIB_FOUND([Libs="$F77LIBS"])
dnl         FEATURE_ENABLE
dnl       ],[
dnl         FEATURE_DISABLE
dnl       ])
dnl     fi
dnl   ])
dnl ACX_LIB_LINK([KAAPIFLIB], [FORTRAN])

dnl -----------------------------------------------------------------------
dnl The "--enable-api-kaapixx"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-kaapixx], [compile Kaapi++ API],
  [Kaapi++ API compiled], [check],
  [],
  [],
  [KAAPI_API_KAAPIXX],
  [
    if test "$have_cxx" = no; then
      AC_MSG_WARN([a C++ compiler is required for Kaapi++ API])
      FEATURE_DISABLE
    else
      FEATURE_ENABLE
    fi
  ])


dnl -----------------------------------------------------------------------
dnl The "--enable-api-quark"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-quark], [compile Quark API used by PLASMA library],
  [Quark API compiled], [yes],
  [],
  [],
  [KAAPI_API_QUARK])


dnl -----------------------------------------------------------------------
dnl The "--enable-api-atha"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-atha], [compile Athapascan-1 API, [[DEPRECATED]] ],
  [Athapascan-1 API compiled], [no],
  [],
  [],
  [KAAPI_API_ATHA],
  [
    if test "$enable_api_kaapixx" != yes ; then
      AC_MSG_NOTICE([Kaapi++ API is required for Athapascan-1 API])
      FEATURE_DISABLE
    else
      FEATURE_ENABLE
    fi
  ])


dnl -----------------------------------------------------------------------
dnl The "--enable-api-kastl"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([api-kastl], [compile KaSTL API],
  [KaSTL API compiled], [no],
  [],
  [],
  [KAAPI_API_KASTL],
  [
    if test "$have_cxx" = no; then
      AC_MSG_WARN([a C++ compiler is required for KaSTL API])
      FEATURE_DISABLE
    else
      FEATURE_ENABLE
    fi
  ])

dnl -----------------------------------------------------------------------
dnl The "--enable-libgomp"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([libgomp], [compile OpenMP libgomp library],
  [OpenMP libgomp compiled], [check],
  [],
  [],
  [KAAPI_LIBGOMP],
  [
    if test "$enable_api_kaapic" != yes ; then
      AC_MSG_NOTICE([KaapiC API is required for OpenMP libgomp])
      FEATURE_DISABLE
    else
      FEATURE_ENABLE
    fi
  ])


dnl ----------------------------------------------------------------
dnl Compilation options
dnl ----------------------------------------------------------------
# Select which target is built.
# Default to mt
AC_ARG_ENABLE([target],[AS_HELP_STRING([--enable-target=<target>],
	[choose target hardware (mt*,gpu,mpsoc-st)])],[],
  [enable_target=default]
)
AS_CASE([$enable_target],
  [mt|MT|default], [dnl
    AC_MSG_NOTICE([Using target MT])
    KAAPI_TARGET=mt
    AC_DEFINE([KAAPI_TARGET_MT],1,[defined to 1 when compiling for multiprocessor target])
  ],[gpu|GPU|mpsoc-st|MPSOC-ST], [dnl
    AC_MSG_ERROR([Using target '$enable_target' not yet available])
  ],[dnl
    AC_MSG_ERROR([Unknown target '$enable_target'. Aborting.])
  ])
AC_SUBST([KAAPI_TARGET])
AM_CONDITIONAL([KAAPI_TARGET_MT],[test x"$KAAPI_TARGET" = x"mt"])


dnl ----------------------------------------------------------------
# Debug mode, CAS and perfcounter
AC_ARG_ENABLE([mode],[AS_HELP_STRING([--enable-mode],
		 [choose special compilation mode (debug, release)])],[],
  [enable_mode=default]
)
AS_CASE([$enable_mode],
  [default], [dnl
  ], [debug], [dnl
    AC_MSG_NOTICE([Using debug compile mode])
    # not using AC*DEFINE as we want this available for examples too
    AM_CPPFLAGS="$AM_CPPFLAGS -DKAAPI_DEBUG"
    CFLAGS=""
    CXXFLAGS=""
    AM_CFLAGS="$AM_CFLAGS -g"
    AM_CXXFLAGS="$AM_CXXFLAGS -g"
  ], [release], [dnl
    AC_MSG_NOTICE([Using release compile mode])
    # using CFLAGS and not AM_CFLAGS as the user must be able to overwrite
    # them on the make command line
    AM_CPPFLAGS="$AM_CPPFLAGS"
    CFLAGS=""
    CXXFLAGS=""
    AM_CFLAGS="$AM_CFLAGS -O3 -DNDEBUG"
    AM_CXXFLAGS="$AM_CXXFLAGS -O3 -DNDEBUG"
  ], [dnl
    AC_MSG_ERROR([Unknown compilation mode '$enable_mode'. Aborting.])
  ])

dnl ----------------------------------------------------------------
AC_ARG_WITH([perfcounter],[AS_HELP_STRING([--with-perfcounter],
		 [add perfcounter support])],[],
  [with_perfcounter=default]
)
AS_CASE([$with_perfcounter],
  [default|no], [dnl
  ], [yes], [dnl
    AC_MSG_NOTICE([Adding perfcounter support])
    # not using AC*DEFINE as we want this available for examples too
    AM_CPPFLAGS="$AM_CPPFLAGS -DKAAPI_USE_PERFCOUNTER=1"
  ], [dnl
    AC_MSG_ERROR([Unknown value '$with_perfcounter' for --with-perfcounter option. Aborting.])
  ])


# Select compilation of static scheduling as part of the library
# Default is to compile it
AC_ARG_ENABLE([partitioning],[AS_HELP_STRING([--enable-partitioning],
	[Disable to not compile static scheduling code, default: disable])],[],
  [enable_partitioning=no]
)
AS_IF([test "x$enable_partitioning" != "xno"], 
  [dnl Do the stuff needed for enabling the feature
    AC_MSG_NOTICE([Compiling static scheduling])
    KAAPI_USE_STATICSCHED=yes
    AC_DEFINE([KAAPI_USE_STATICSCHED],1,[defined to 1 when compiling with static scheduling])
  ],
 
  [dnl Do the stuff needed for disable the feature
    AC_MSG_NOTICE([Do not compile static scheduling])
  ]
)
AC_SUBST([KAAPI_USE_STATICSCHED])
AM_CONDITIONAL([KAAPI_TARGET_STATICSCHED],[test x"$KAAPI_USE_STATICSCHED" = x"yes"])


dnl ----------------------------------------------------------------
dnl Checks for header files.
dnl ----------------------------------------------------------------
AC_CHECK_HEADERS([arpa/inet.h limits.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])
AC_CHECK_HEADERS([errno.h limits.h numa.h pthread.h sched.h sys/sysctl.h])

# Context switch
AC_CHECK_HEADERS([setjmp.h ucontext.h])

# For debug mode
AC_CHECK_HEADERS([stdio.h])

# Maximum number of architecture depending on the installation (1 or 2 in case
# of mixing CPU/GPU, 3 in case of mixing CPU/GPU/MPSoC)
# This number may be also depend on the caracteristic of the processors.... and
# one could consider having N different architectures
# For each architecture, task should have a specific entry point
# This will lead to incompatible binary library and should be visible in the
# library name
AC_DEFINE([KAAPI_MAX_ARCH],3,[Maximum number of architectures])

KAAPI_FEATURE([max-processors], [Maximum number of processors (special values "host" to optimize for current host and "generic" for runtime-choice)], [], [host],
  [], [], [],
  [ 
  AS_CASE(["$enable_max_processors"],
    [host], [
      AC_MSG_CHECKING([how many processors on this system])
      value=`getconf _NPROCESSORS_ONLN 2>/dev/null`
      if test x"$value" = x; then
        AC_MSG_RESULT([Unknown (using default value)])
  	    FEATURE_SET(["generic"])
      else
        AC_MSG_RESULT([$value])
  	    FEATURE_SET(["$value"])
      fi
    ])
  ],
  [
    FEATURE_CASE_VALUE([no,*], [enable_max_processors=0])
#    FEATURE_CASE_VALUE([generic,*], [enable_max_processors=0])
    FEATURE_CASE_VALUE([generic,*], [AC_MSG_ERROR([Option "generic" is under progress])])
    FEATURE_CASE_VALUE([no,*|yes,*], [AC_MSG_ERROR([A numeric value (or "host" or "generic") is required for --enable-max-processors])])
    FEATURE_CASE_VALUE([[[1-9]]*], [])
  ])
if test "$enable_max_processors" = 0 ; then
  AC_MSG_NOTICE([Maximum number of usable processors... generic])
else
  AC_MSG_NOTICE([Maximum number of usable processors... $enable_max_processors])
fi
AC_DEFINE_UNQUOTED([KAAPI_MAX_PROCESSOR],[$enable_max_processors],[Maximum number of processors (0 for generic implementation)])

dnl 
KAAPI_FEATURE([futex], [use low-level futex synchronisation primitives],
  [using futex], [check],
  [HAVE_FUTEX=1],
  [HAVE_FUTEX=0],
  [],
  [
    AC_CHECK_HEADER([linux/futex.h], [ FEATURE_ENABLE ], [ FEATURE_DISABLE ])
  ])
AC_DEFINE_UNQUOTED([HAVE_FUTEX],[$HAVE_FUTEX],[use futex synchronisation primitives])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_OFF_T
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_TYPE_MODE_T
AC_CHECK_TYPES([ptrdiff_t])

AC_CHECK_SIZEOF([void*])


dnl ----------------------------------------------------------------
dnl Checks for library functions.
dnl ----------------------------------------------------------------
AC_FUNC_ALLOCA
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_FUNC_REALLOC
AC_CHECK_FUNCS([bzero clock_gettime getcwd gethostname getpagesize gettimeofday memset mkdir munmap sqrt strdup strerror strtoul])

AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK

dnl ----------------------------------------------------------------
dnl Checks for libraries.
dnl ----------------------------------------------------------------

# Linkage is done with gcc not g++, so no need to workaround libtool bug
# See http://bugs.debian.org/468555
ACX_LIB_CHECK_PLAIN([PTHREAD],[],[
  ACX_PTHREAD([
    ACX_LIB_FOUND([CppFlags="$PTHREAD_CFLAGS"], [Libs="$PTHREAD_LIBS"], [LdFlags="$PTHREAD_CFLAGS"])
    CC="$PTHREAD_CC"
  ], [
    ACX_LIB_NOTFOUND
  ])
])
ACX_LIB_LINK([KAAPILIB],[PTHREAD])

# search pthread_attr_setaffinity_np within libpthread
#saveCFLAGS="$CFLAGS"
#saveLIBS="$LIBS"
#CFLAGS="$CFLAGS $PTHREAD_CPPFLAGS $PTHREAD_CFLAGS"
#LIBS="$LIBS $PTHREAD_LDFLAGS $PTHREAD_LIBS"
#AC_CHECK_FUNCS([pthread_attr_setaffinity_np],
#  [AC_DEFINE([KAAPI_USE_SCHED_AFFINITY],1,[Use affinity interface])])
#CFLAGS="$saveCFLAGS"
#LIBS="$saveLIBS"

dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl --with && --enable OPTIONS
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------

dnl -----------------------------------------------------------------------
dnl The "--enable-aggregation"
dnl -----------------------------------------------------------------------
# Select compilation of aggregation support
# Default is yes
KAAPI_FEATURE([aggregation], [build with support for work steal requests aggregation strategy],
  [Work steal requests aggregation], [yes],
  [
    AC_DEFINE([KAAPI_USE_AGGREGATION],1,[defined to 1 when compiling with aggregation support])
    KAAPI_USE_AGGREGATION=yes
  ],
  []
  )


dnl -----------------------------------------------------------------------
dnl The "--enable-pthreadaffinity"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([pthreadaffinity], [build with Pthread affinity support (pthread_attr_setaffinity_np)], [], [check],
  [
    AC_DEFINE([KAAPI_USE_SCHED_AFFINITY],1,[Use affinity interface])
  ], [], [],
  [ 
    # search pthread_attr_setaffinity_np within libpthread
    saveCFLAGS="$CFLAGS"
    saveLIBS="$LIBS"
    CFLAGS="$CFLAGS $PTHREAD_CPPFLAGS $PTHREAD_CFLAGS"
    LIBS="$LIBS $PTHREAD_LDFLAGS $PTHREAD_LIBS"
    AC_CHECK_FUNCS([pthread_attr_setaffinity_np],
      [AC_DEFINE([KAAPI_USE_SCHED_AFFINITY],1,[Use affinity interface])])
    CFLAGS="$saveCFLAGS"
    LIBS="$saveLIBS"
  ])

dnl -----------------------------------------------------------------------
dnl The "--with-poti"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([POTI], [poti], [POTI], [no], [KAAPI_USE_EXTERNAL_POTI],
  [use POTI library to produce Paje traces],
  [poti], [], [poti.h], [pajeOpen])

dnl -----------------------------------------------------------------------
dnl The "--with-papi"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([PAPI], [papi], [PAPI], [no], [],
  [build PAPI perfcounters [EXPERIMENTAL]],
  [papi], [], [papi.h], [PAPI_library_init], [
    dnl PAPI is there
    AC_DEFINE(KAAPI_USE_PAPI, 1, [Define to use PAPI perfcounters])
    AM_CPPFLAGS="$AM_CPPFLAGS -DKAAPI_USE_PAPI=1"
  ])
ACX_LIB_LINK([KAAPILIB],[PAPI])


dnl -----------------------------------------------------------------------
dnl The "--with-numa"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([NUMA extensions], [numa], [NUMA], [no], [KAAPI_USE_NUMA],
  [build with numa extensions [EXPERIMENTAL]],
  [numa], [], [numa.h], [numa_bind], [
    dnl NUMA is there
    AC_DEFINE(KAAPI_USE_NUMA, 1, [Define to use memory affinity on numa nodes])
    dnl ACX_LIB_ADD_PUBLIC_FLAGS([KAAPILIB], [-lnuma])
  ])
ACX_LIB_LINK([KAAPILIB],[NUMA])


dnl -----------------------------------------------------------------------
dnl The "--with-cuda"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([GPU programming], [cuda], [CUDA], [no], [KAAPI_USE_CUDA],
  [build CUDA support [EXPERIMENTAL]],
  [cuda], [], [cuda_runtime_api.h], [cuDeviceGetAttribute], [
dnl    CUDA is there
    AC_DEFINE([KAAPI_USE_CUDA],1,[Support for CUDA])
    HACK_USE_CUDA=yes
  ],
  [
  ],
  [-lcublas])
AM_CONDITIONAL([KAAPI_TARGET_CUDA],[test x"$HACK_USE_CUDA" = x"yes"])
ACX_LIB_LINK([KAAPILIB],[CUDA])

dnl -----------------------------------------------------------------------
dnl The "--with-cupti"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([GPU programming], [cupti], [CUPTI], [no], [KAAPI_USE_CUPTI],
  [build CUDA profiler support [EXPERIMENTAL]],
  [cupti], [], [cuda_runtime_api.h], [cuDeviceGetAttribute], [
dnl    CUPTI is there
    AC_DEFINE([KAAPI_USE_CUPTI],1,[Support for CUDA profiler])
    HACK_USE_CUPTI=yes
  ],
  [
  ],
  [])
AM_CONDITIONAL([KAAPI_TARGET_CUPTI],[test x"$HACK_USE_CUPTI" = x"yes"])
ACX_LIB_LINK([KAAPILIB],[CUPTI])


dnl -----------------------------------------------------------------------
dnl The "--with-hwloc"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([hwloc], [build with HWloc support for discover hardware topology], [], [check],
  [
    AC_DEFINE([KAAPI_USE_HWLOC],1,[Use HWLoc to discover the topology])
  ], [], [],
  [ 
    KAAPI_PKGLIB_CHECK([HWLOC], [hwloc], [HWLOC], [], [KAAPI_USE_HWLOC],
     [build with HWloc support for discover hardware topology],
     [hwloc], [], [hwloc.h], [hwloc_topology_init], [
       FEATURE_ENABLE
     ],
     [
       FEATURE_DISABLE
     ],
     [])

  ])
ACX_LIB_LINK([KAAPILIB],[HWLOC])

dnl -----------------------------------------------------------------------
dnl The "--enable-network"
dnl -----------------------------------------------------------------------
# Select compilation of network support as part of the library
# Default is to not compile it until it works
KAAPI_FEATURE([network], [allow distributed Kaapi programs],
  [Network support compiled], [no],
  [
    AC_DEFINE([KAAPI_USE_NETWORK],1,[defined to 1 when compiling with network support])
    KAAPI_USE_NETWORK=yes
  ],
  [],
  [KAAPI_TARGET_KANET],
  [
    ## Ici, il faudrait tester les r√seaux possibles et les souhait
    # de l'utilisateur.
    FEATURE_ENABLE
  ])

## ICI il faudrait preciser le reseau dans le commande --with-network=...
## sinon choix par d√©faut
AM_CONDITIONAL([KAAPI_TARGET_MPINET],[test x"$KAAPI_USE_NETWORK" = x"yes"])



dnl -----------------------------------------------------------------------
dnl The "--with-gasnet"
dnl -----------------------------------------------------------------------
AC_ARG_WITH([gasnet],[AS_HELP_STRING([--with-gasnet=<conduit>],
		 [choose gasnet support and specify the conduit to use])],
     [],
     [with_gasnet=no]
)

AS_CASE([$with_gasnet],
  [no], [dnl
  ], 
  [*], [dnl
    AC_MSG_NOTICE([Adding gasnet support for conduit '$with_gasnet'])
    # not using AC*DEFINE as we want this available for examples too
    AC_DEFINE([KAAPI_USE_GASNET],1,[Use with GASNet])
    KAAPI_USE_GASNET=yes
    tmpfilename=`mktemp -t kaapi_XXXXXX`
    make -f $srcdir/acinclude/gasnet_cppflags.mk GASNET_CONDUIT_MAK=$with_gasnet > $tmpfilename
    source $tmpfilename
    KAGASNETLIB_BUILD_CPPFLAGS=$GASNET_CPPFLAGS
    KAGASNETLIB_BUILD_CFLAGS=$GASNET_CFLAGS
    KAGASNETLIB_BUILD_LDFLAGS=$GASNET_LDFLAGS
    KAGASNETLIB_BUILD_LIBS="$GASNET_LIBS"
    AC_MSG_NOTICE([KAGASNETLIB_BUILD_CPPFLAGS=$KAGASNETLIB_BUILD_CPPFLAGS])
    AC_MSG_NOTICE([KAGASNETLIB_BUILD_CFLAGS=$KAGASNETLIB_BUILD_CFLAGS])
    AC_MSG_NOTICE([KAGASNETLIB_BUILD_LDFLAGS=$KAGASNETLIB_BUILD_LDFLAGS])
    AC_MSG_NOTICE([KAGASNETLIB_BUILD_LIBS=$KAGASNETLIB_BUILD_LIBS])
  ]
)
AC_SUBST(KAGASNETLIB_BUILD_CPPFLAGS)
AC_SUBST(KAGASNETLIB_BUILD_CFLAGS)
AC_SUBST(KAGASNETLIB_BUILD_LDFLAGS)
AC_SUBST(KAGASNETLIB_BUILD_LIBS)
AM_CONDITIONAL([KAAPI_TARGET_GASNET],[test x"$KAAPI_USE_GASNET" = x"yes"])




dnl -----------------------------------------------------------------------
dnl The "--with-taknet"
dnl -----------------------------------------------------------------------
AC_ARG_WITH([taknet],[AS_HELP_STRING([--with-taknet=<conduit>],
		 [choose taknet Network support based on TakTuk])],
     [],
     [with_taknet=no]
)

AS_CASE([$with_taknet],
  [no], [dnl
  ], 
  [*], [dnl
    AC_MSG_NOTICE([Adding taknet support '$with_taknet'])
    # not using AC*DEFINE as we want this available for examples too
    AC_DEFINE([KAAPI_USE_TAKNET],1,[Use with GASNet])
    KAAPI_USE_TAKNET=yes
    KATAKNETLIB_BUILD_CPPFLAGS=-I$with_taknet/include
    KATAKNETLIB_BUILD_CFLAGS=
    KATAKNETLIB_BUILD_LDFLAGS=-L$with_taknet/lib
    KATAKNETLIB_BUILD_LIBS="-ltaktuk"
    AC_MSG_NOTICE([KATAKNETLIB_BUILD_CPPFLAGS=$KATAKNETLIB_BUILD_CPPFLAGS])
    AC_MSG_NOTICE([KATAKNETLIB_BUILD_CFLAGS=$KATAKNETLIB_BUILD_CFLAGS])
    AC_MSG_NOTICE([KATAKNETLIB_BUILD_LDFLAGS=$KATAKNETLIB_BUILD_LDFLAGS])
    AC_MSG_NOTICE([KATAKNETLIB_BUILD_LIBS=$KATAKNETLIB_BUILD_LIBS])
  ]
)
AC_SUBST(KATAKNETLIB_BUILD_CPPFLAGS)
AC_SUBST(KATAKNETLIB_BUILD_CFLAGS)
AC_SUBST(KATAKNETLIB_BUILD_LDFLAGS)
AC_SUBST(KATAKNETLIB_BUILD_LIBS)
AM_CONDITIONAL([KAAPI_TARGET_TAKNET],[test x"$KAAPI_USE_TAKNET" = x"yes"])


dnl -----------------------------------------------------------------------
dnl The "--with-boost" required for kacc compiler with rose
dnl -----------------------------------------------------------------------
AC_ARG_WITH([boost],[AS_HELP_STRING([--with-boost=<dir>],
		 [choose boost directory. required for if enable kaapi compiler])],
     [],
     [with_boost=no]
)

AS_CASE([$with_boost],
  [no], [dnl
  ], 
  [*], [dnl
    AC_MSG_NOTICE([Adding boost support '$with_boost'])
    # not using AC*DEFINE as we want this available for examples too
    KAAPI_USE_BOOST=yes
    BOOST_BUILD_CPPFLAGS=-I$with_boost/include
    BOOST_BUILD_CFLAGS=
    BOOST_BUILD_LDFLAGS=-L$with_boost/lib
    BOOST_BUILD_LIBS=
    AC_MSG_NOTICE([BOOST_BUILD_CPPFLAGS=$BOOST_BUILD_CPPFLAGS])
    AC_MSG_NOTICE([BOOST_BUILD_CFLAGS=$BOOST_BUILD_CFLAGS])
    AC_MSG_NOTICE([BOOST_BUILD_LDFLAGS=$BOOST_BUILD_LDFLAGS])
    AC_MSG_NOTICE([BOOST_BUILD_LIBS=$BOOST_BUILD_LIBS])
  ]
)
AC_SUBST(BOOST_BUILD_CPPFLAGS)
AC_SUBST(BOOST_BUILD_CFLAGS)
AC_SUBST(BOOST_BUILD_LDFLAGS)
AC_SUBST(BOOST_BUILD_LIBS)

dnl -----------------------------------------------------------------------
dnl The "--with-java" required for kacc compiler with rose
dnl -----------------------------------------------------------------------
AC_ARG_WITH([java],[AS_HELP_STRING([--with-java=<dir>],
		 [choose java directory. required for if enable kaapi compiler])],
     [],
     [with_java=no]
)

AS_CASE([$with_java],
  [no], [dnl
  ], 
  [*], [dnl
    AC_MSG_NOTICE([Adding java support '$with_java'])
    # not using AC*DEFINE as we want this available for examples too
    KAAPI_USE_JAVA=yes
    JAVA_BUILD_LDFLAGS=-L$with_java
    JAVA_BUILD_LIBS=-ljvm
    AC_MSG_NOTICE([JAVA_BUILD_LDFLAGS=$JAVA_BUILD_LDFLAGS])
    AC_MSG_NOTICE([JAVA_BUILD_LIBS=$JAVA_BUILD_LIBS])
  ]
)
AC_SUBST(JAVA_BUILD_LDFLAGS)
AC_SUBST(JAVA_BUILD_LIBS)


dnl -----------------------------------------------------------------------
dnl The "--with-rose"
dnl -----------------------------------------------------------------------
AC_ARG_WITH([rose],[AS_HELP_STRING([--with-rose=<dir>],
		 [choose Rose (http://www.rosecompiler.org/) installation directory to compile kacc])],
     [],
     [with_rose=no]
)

AS_CASE([$with_rose],
  [no], [dnl
  ], 
  [*], [dnl
    AC_MSG_NOTICE([Adding Rose '$with_rose' for compiler generation])
    # not using AC*DEFINE as we want this available for examples too
    KAAPI_USE_ROSE=yes
    ROSE_BUILD_CPPFLAGS="-I$with_rose/include $BOOST_BUILD_CPPFLAGS"
    ROSE_BUILD_CFLAGS=
    ROSE_BUILD_LDFLAGS="-L$with_rose/lib $BOOST_BUILD_LDFLAGS $JAVA_BUILD_LDFLAGS"
    ROSE_BUILD_LIBS="-framework JavaVM -L$with_rose/lib -lrose $ROSE_BUILD_LDFLAGS  $BOOST_BUILD_LDFLAGS -lm -lboost_date_time -lboost_thread -lboost_filesystem -lboost_program_options -lboost_regex -lboost_system -lboost_wave -ldl $JAVA_BUILD_LDFLAGS $JAVA_BUILD_LIBS"
    AC_MSG_NOTICE([ROSE_BUILD_CPPFLAGS=$ROSE_BUILD_CPPFLAGS])
    AC_MSG_NOTICE([ROSE_BUILD_CFLAGS=$ROSE_BUILD_CFLAGS])
    AC_MSG_NOTICE([ROSE_BUILD_LDFLAGS=$ROSE_BUILD_LDFLAGS])
    AC_MSG_NOTICE([ROSE_BUILD_LIBS=$ROSE_BUILD_LIBS])
  ]
)
AM_CONDITIONAL([KAAPI_TARGET_ROSE],[test x"$KAAPI_USE_ROSE" = x"yes"])


dnl -----------------------------------------------------------------------
dnl The "--enable-kacc"
dnl -----------------------------------------------------------------------
KAAPI_FEATURE([kacc], [Kaapi C/C++ compiler],
  [KaCC generated], [no],
  [
    AC_DEFINE([KAAPI_USE_KACC],1,[defined to 1 when compiling with kacc support])
    KAAPI_USE_KACC=yes
    KACC_BUILD_CPPFLAGS="$ROSE_BUILD_CPPFLAGS $BOOST_BUILD_CPPFLAGS"
    KACC_BUILD_CFLAGS="$ROSE_BUILD_CFLAGS $BOOST_BUILD_CFLAGS"
    KACC_BUILD_LDFLAGS="$ROSE_BUILD_LDFLAGS"
    KACC_BUILD_LIBS="$ROSE_BUILD_LIBS"
    AC_MSG_NOTICE([KACC_BUILD_CPPFLAGS=$KACC_BUILD_CPPFLAGS])
    AC_MSG_NOTICE([KACC_BUILD_CFLAGS=$KACC_BUILD_CFLAGS])
    AC_MSG_NOTICE([KACC_BUILD_LDFLAGS=$KACC_BUILD_LDFLAGS])
    AC_MSG_NOTICE([KACC_BUILD_LIBS=$KACC_BUILD_LIBS])
  ],
  [],
  [KAAPI_TARGET_KACC],
  [
    FEATURE_ENABLE
  ])

AC_SUBST(KACC_BUILD_CPPFLAGS)
AC_SUBST(KACC_BUILD_CFLAGS)
AC_SUBST(KACC_BUILD_LDFLAGS)
AC_SUBST(KACC_BUILD_LIBS)
AM_CONDITIONAL([KAAPI_TARGET_KACC],[test x"$KAAPI_USE_KACC" = x"yes"])


AM_CONDITIONAL([KAAPI_HAVE_CBLAS],[test x"$CBLAS_LDFLAGS" != x])

dnl -----------------------------------------------------------------------
dnl Dependencies between libs
dnl -----------------------------------------------------------------------
ACX_LIB_LINK([KAAPIXXLIB], [PUBLIC], [KAAPILIB])
ACX_LIB_LINK([KAAPIXXLIB], [PRIVATE], [CUDA])
ACX_LIB_LINK([KAAPIXXLIB], [PRIVATE], [CUPTI])
dnl ACX_LIB_LINK([KAAPIXXLIB], [PRIVATE], [HWLOC])

dnl ACX_LIB_LINK([KANETLIB], [PUBLIC], [KAAPIXXLIB])
ACX_LIB_LINK([KANETLIB], [PRIVATE], [KAAPIXXLIB])

ACX_LIB_LINK([ATHALIB], [PUBLIC], [KAAPIXXLIB])
ACX_LIB_LINK([ATHALIB], [PRIVATE], [CUDA])
ACX_LIB_LINK([ATHALIB], [PRIVATE], [CUPTI])
dnl ACX_LIB_LINK([ATHALIB], [PRIVATE], [HWLOC])

ACX_LIB_LINK([KASTLLIB], [PUBLIC], [KAAPIXXLIB])
ACX_LIB_LINK([KASTLLIB], [PRIVATE], [CUDA])
ACX_LIB_LINK([KASTLLIB], [PRIVATE], [CUPTI])
dnl ACX_LIB_LINK([KASTLLIB], [PRIVATE], [HWLOC])

ACX_LIB_LINK([KAAPICLIB], [PUBLIC], [KAAPILIB])
ACX_LIB_LINK([KAAPICLIB], [PRIVATE], [CUDA])
ACX_LIB_LINK([KAAPICLIB], [PRIVATE], [CUPTI])

ACX_LIB_LINK([KAAPIFLIB], [PUBLIC], [KAAPICLIB])
ACX_LIB_LINK([KAAPIFLIB], [PRIVATE], [CUDA])
ACX_LIB_LINK([KAAPIFLIB], [PRIVATE], [CUPTI])
dnl ACX_LIB_LINK([KAAPIFLIB], [PRIVATE], [HWLOC])

ACX_LIB_LINK([LIBGOMP], [PUBLIC], [KAAPILIB KAAPICLIB])
ACX_LIB_LINK([LIBGOMP], [PRIVATE], [KAAPILIB KAAPICLIB CUDA])

ACX_LIB_LINK([KAAPIPROGS], [KAAPILIB])
ACX_LIB_ADD_BUILD_FLAGS([KAAPIPROGS],
	[-I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/src/libkaapi.la])
ACX_LIB_LINK([KAAPIPROGS], [CUDA])
ACX_LIB_LINK([KAAPIPROGS], [CUPTI])

ACX_LIB_LINK([KAAPICPROGS], [KAAPICLIB])
ACX_LIB_ADD_BUILD_FLAGS([KAAPICPROGS],
	[-I\$(top_srcdir)/api/kaapic -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/api/kaapic/libkaapic.la \$(top_builddir)/src/libkaapi.la])

ACX_LIB_LINK([KAAPIFPROGS], [KAAPIFLIB])
ACX_LIB_ADD_BUILD_FLAGS([KAAPIFPROGS],
	[-I\$(top_srcdir)/api/kaapif -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/api/kaapif/libkaapif.la \$(top_builddir)/api/kaapic/libkaapic.la \$(top_builddir)/src/libkaapi.la])

ACX_LIB_LINK([KAAPIXXPROGS], [KAAPIXXLIB])
ACX_LIB_ADD_BUILD_FLAGS([KAAPIXXPROGS],
	[-I\$(top_srcdir)/api/kaapixx -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/api/kaapixx/libkaapi++.la \$(top_builddir)/src/libkaapi.la])

ACX_LIB_LINK([KANETPROGS], [KANETLIB])
ACX_LIB_ADD_BUILD_FLAGS([KANETPROGS],
	[-I\$(top_srcdir)/network -I\$(top_srcdir)/api/kaapixx -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/network/libkanet.la \$(top_builddir)/api/kaapixx/libkaapi++.la \$(top_builddir)/src/libkaapi.la])

ACX_LIB_LINK([ATHAPROGS], [ATHALIB])
ACX_LIB_ADD_BUILD_FLAGS([ATHAPROGS],
	[-I\$(top_srcdir)/api/atha -I\$(top_srcdir)/api/kaapixx -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/api/atha/libatha.la \$(top_builddir)/api/kaapixx/libkaapi++.la \$(top_builddir)/src/libkaapi.la])

ACX_LIB_LINK([KASTLPROGS], [KASTLLIB])
ACX_LIB_ADD_BUILD_FLAGS([KASTLPROGS],
	[-I\$(top_srcdir)/api/kastl -I\$(top_srcdir)/api/kaapixx -I\$(top_srcdir)/src],[],[],
	[\$(top_builddir)/api/kaapixx/libkaapi++.la \$(top_builddir)/src/libkaapi.la])
dnl	libkastl.la should be added if libs will be created \$(top_builddir)/api/kastl/libkastl.la 


AC_SUBST(AM_CPPFLAGS)
AC_SUBST(AM_CFLAGS)
AC_SUBST(AM_CXXFLAGS)

AM_LDFLAGS="$LDFLAGS"
AC_SUBST(AM_LDFLAGS)

dnl First, alphabetic order of directory of Makefiles
dnl Then   alphabetic order of other files
AC_CONFIG_FILES([
	Makefile
	api/Makefile
	api/atha/Makefile
	api/kaapixx/Makefile
	api/kaapic/Makefile
	api/kaapif/Makefile
	api/kastl/Makefile
	api/libgomp/Makefile
	api/libgomp/tests/Makefile
	api/libgomp/testsuite/Makefile
	api/quark/Makefile
	examples/Makefile
	network/Makefile
	network/gasnet/Makefile
	network/taknet/Makefile
	src/Makefile
	bin/Makefile
	bin/kacc/Makefile
	bin/kaapi_c2c/Makefile
	bin/kaapi_f2f/Makefile
	bin/kaapi_event_reader/Makefile
	bin/kacc/kacc.cpp
	tests/static/Makefile
	tests/c/Makefile
	tests/c++/Makefile

	src/misc/kaapi_version.c
	kaapi.pc
	kaapic.pc
	kaapif.pc
	kaapi++.pc
	kastl.pc
	atha.pc
])
AC_OUTPUT

AC_MSG_NOTICE([*************************************])
AC_MSG_NOTICE([*             Summary               *])
AC_MSG_NOTICE([*************************************])
ACX_LIB_SHOW_EXTERNAL_LIBS
AC_MSG_NOTICE([*************************************])
ACX_LIB_SHOW_NEW_LIBS
AC_MSG_NOTICE([*************************************])
ACX_LIB_SHOW_NEW_PRGS
AC_MSG_NOTICE([*************************************])
list=""
for flag in CC CXX CFLAGS CPPFLAGS CXXFLAGS; do
  for f in AM@&t@_$flag $flag ; do
    val=`eval echo '"$'$f'"'`
    val=`echo "$val" | sed 's/^ *//'`
    if test x"$val" != x ; then
       list="$list
  $f: $val"
    fi
  done
done
AC_MSG_NOTICE([Compilers and default flags used:$list])
AC_MSG_NOTICE([*************************************])
KAAPI_FEATURE_RESUME
AC_MSG_NOTICE([*************************************])
