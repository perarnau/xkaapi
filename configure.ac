#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.60])

AC_INIT([X-KAAPI], [0.1], [kaapi-dev-info@lists.gforge.inria.fr],[xkaapi])
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
dnl or in the build tree
if test -d "$prefix" ; then
   prefix_pwd=`cd $prefix ; pwd`
   if test x"$prefix_pwd" = x`test -n "$srcdir" && cd "$srcdir" ; pwd` ; then
      AC_MSG_WARN([Invalid installation prefix '$prefix':])
      AC_MSG_ERROR([X-KAAPI cannot be installed at the root of the source directory])
   fi
   if test x"$prefix_pwd" = x`test -n "$builddir" && cd "$builddir" ; pwd` ; then
      AC_MSG_WARN([Invalid installation prefix '$prefix':])
      AC_MSG_ERROR([X-KAAPI cannot be installed at the root of the build directory])
   fi
fi

dnl do not change order of --with and --enable options
AC_PRESERVE_HELP_ORDER

dnl ----------------------------------------------------------------
dnl Checks for programs.
dnl ----------------------------------------------------------------
PKG_PROG_PKG_CONFIG
AC_PROG_CXX
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_INSTALL
ACX_PROG_CCACHE_CC
ACX_PROG_CCACHE_CXX

# if next line is comented out, we can create only dynamic libraries be default
# AC_DISABLE_STATIC
AC_PROG_LIBTOOL
AC_SUBST(LIBTOOL_DEPS)

AC_PROG_MKDIR_P

AC_CACHE_SAVE

NESTED_FUNCTIONS_CFLAGS=""
AX_CHECK_COMPILER_FLAGS([-fnested-functions],
  [NESTED_FUNCTIONS_CFLAGS="-fnested-functions"])
AC_SUBST([NESTED_FUNCTIONS_CFLAGS])


AX_CHECK_COMPILER_STD([STD_C99], [C], [-std=c99], [], [
   STD_C99_FLAGS="-D_KAAPI_DISABLE_WARNINGS"
   AC_MSG_WARN([Unable to select C99 standard with the current C compiler.])
   AC_MSG_WARN([Pedantic mode is disabled and some warning and/or errors can occur])
])
AX_CHECK_COMPILER_STD([STD_CXX0X], [C++], [-std=c++0x], [], [
   STD_CXX0X_FLAGS="-D_KAAPI_DISABLE_WARNINGS"
   AC_MSG_WARN([Unable to select C++0X standard with the current C++ compiler.])
   AC_MSG_WARN([Pedantic mode is disabled and some warning and/or errors can occur])
])


ACX_LIB_NEW_LIB([XKAAPILIB], [-L\${libdir} -lxkaapi], [xkaapi])
ACX_LIB_ADD_PUBLIC_FLAGS([XKAAPILIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([XKAAPIPPLIB], [-L\${libdir} -lxkaapi++], [xkaapi++])
ACX_LIB_ADD_PUBLIC_FLAGS([XKAAPIPPLIB], [-I\${includedir}])

ACX_LIB_NEW_PRG([XKAAPIPROGS])
ACX_LIB_NEW_PRG([XKAAPIPPPROGS])

#ACX_LIB_NEW_LIB([ATHA1], [-L\${libdir} -latha1], [atha1])
#ACX_LIB_ADD_PUBLIC_FLAGS([ATHA1], [-I\${includedir}])

#ACX_LIB_NEW_PRG([ATHA1PROGS])

# add warning flags to the C and C++ compilers
AX_CFLAGS_WARN_ALL([AM_CFLAGS])
AX_CXXFLAGS_WARN_ALL([AM_CXXFLAGS])

dnl ----------------------------------------------------------------
dnl Force flag to be empty...
dnl ----------------------------------------------------------------


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
    AM_CFLAGS="$AM_CFLAGS -O4 -DNDEBUG"
    AM_CXXFLAGS="$AM_CXXFLAGS -O4 -DNDEBUG"
  ], [dnl
    AC_MSG_ERROR([Unknown compilation mode '$enable_mode'. Aborting.])
  ])

AC_ARG_WITH([steal],[AS_HELP_STRING([--with-steal],
		 [choose the work stealing algorithm (cas,hybrid,disjkstra)])],[],
  [with_steal=hybrid]
)
AS_CASE([$with_steal],
  [hybrid], [dnl
    AC_MSG_NOTICE([configure with hybrid workstealing algorithm])
    # must not change the ABI. Enabling through AC*DEFINE
    # 0 -> KAAPI_STEALCAS_METHOD 1->KAAPI_STEALTHE_METHOD 
    AC_DEFINE([KAAPI_USE_STEALFRAME_METHOD],1,[Support concurrent work stealing/task])
    AC_DEFINE([KAAPI_USE_STEALTASK_METHOD],0,[Support concurrent work stealing/frame])
  ], [cas], [dnl
    AC_MSG_NOTICE([configure with CAS based workstealing algorithm])
    # must not change the ABI. Enabling through AC*DEFINE
    # 0 -> KAAPI_STEALCAS_METHOD 1->KAAPI_STEALTHE_METHOD 
    AC_DEFINE([KAAPI_USE_STEALFRAME_METHOD],0,[Support concurrent work stealing/task])
    AC_DEFINE([KAAPI_USE_STEALTASK_METHOD],0,[Support concurrent work stealing/frame])
  ], [disjkstra], [dnl
    AC_MSG_NOTICE([configure with full Disjkstra based workstealing protocol])
    # must not change the ABI. Enabling through AC*DEFINE
    # 0 -> KAAPI_STEALCAS_METHOD 1->KAAPI_STEALTHE_METHOD 
    AC_DEFINE([KAAPI_USE_STEALFRAME_METHOD],1,[Support concurrent work stealing/task])
    AC_DEFINE([KAAPI_USE_STEALTASK_METHOD],1,[Support concurrent work stealing/frame])
  ], [dnl
   AC_MSG_ERROR([Unknown value '$with_steal' for --with-steal option. Aborting.])
  ])

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

AC_DEFINE([KAAPI_MAX_PROCESSOR],16,[Maximum number of processors])

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
AC_CHECK_TYPES([ptrdiff_t])

AC_CHECK_SIZEOF([void*])

dnl ----------------------------------------------------------------
dnl Checks for library functions.
dnl ----------------------------------------------------------------
AC_FUNC_ALLOCA
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_FUNC_REALLOC
AC_CHECK_FUNCS([clock_gettime getcwd gethostname getpagesize gettimeofday memset mkdir munmap strdup strerror strtoul])

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
ACX_LIB_LINK([XKAAPILIB],[PTHREAD])

# search pthread_attr_setaffinity_np within libpthread
saveCFLAGS="$CFLAGS"
saveLIBS="$LIBS"
CFLAGS="$CFLAGS $PTHREAD_CPPFLAGS $PTHREAD_CFLAGS"
LIBS="$LIBS $PTHREAD_LDFLAGS $PTHREAD_LIBS"
AC_CHECK_FUNCS([pthread_attr_setaffinity_np],
  [AC_DEFINE([KAAPI_USE_SCHED_AFFINITY],1,[Use affinity interface])])
CFLAGS="$saveCFLAGS"
LIBS="$saveLIBS"

dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl --with && --enable OPTIONS
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------

dnl -----------------------------------------------------------------------
dnl The "--with-papi"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([PAPI], [papi], [PAPI], [no], [],
  [build PAPI perfcounters [EXPERIMENTAL]],
  [papi], [], [papi.h], [PAPI_library_init], [
    dnl PAPI is there
    AC_DEFINE(KAAPI_USE_PAPI, 1, [Define to use PAPI perfcounters])
    AM_CPPFLAGS="$AM_CPPFLAGS -DKAAPI_USE_PAPIPERFCOUNTER=1"
  ])
ACX_LIB_LINK([XKAAPILIB],[PAPI])

dnl -----------------------------------------------------------------------
dnl The "--with-numa"
dnl -----------------------------------------------------------------------
KAAPI_PKGLIB_CHECK([NUMA extensions], [numa], [NUMA], [no], [KAAPI_USE_NUMA],
  [build with numa extensions [EXPERIMENTAL]],
  [numa], [], [numa.h], [numa_bind], [
    dnl NUMA is there
    AC_DEFINE(KAAPI_USE_NUMA, 1, [Define to use memory affinity on numa nodes])
  ])
ACX_LIB_LINK([XKAAPILIB],[NUMA])

#ACX_LIB_LINK([ATHA1], [PUBLIC], [XKAAPILIB])
ACX_LIB_LINK([XKAAPIPPLIB], [PUBLIC], [XKAAPILIB])

ACX_LIB_LINK([XKAAPIPROGS], [XKAAPILIB])
#ACX_LIB_LINK([ATHA1PROGS], [ATHA1])
ACX_LIB_LINK([XKAAPIPPPROGS], [XKAAPIPPLIB])

AC_SUBST(AM_CPPFLAGS)
AC_SUBST(AM_CFLAGS)
AC_SUBST(AM_CXXFLAGS)

AM_LDFLAGS="$LDFLAGS"
AC_SUBST(AM_LDFLAGS)

AC_CONFIG_FILES([
	Makefile
	api/Makefile
	examples/Makefile
	src/Makefile

	xkaapi.pc
	atha1.pc
	src/misc/kaapi_version.c
])
#	kastl.wip/Makefile
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

