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
AM_INIT_AUTOMAKE([gnu 1.9 tar-pax subdir-objects silent-rules])

AC_CONFIG_HEADERS([config.h])

AM_SILENT_RULES
AM_DEFAULT_VERBOSITY=0

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

ACX_LIB_NEW_LIB([XKAAPILIB], [-L\${libdir} -lxkaapi], [xkaapi])
ACX_LIB_ADD_PUBLIC_FLAGS([XKAAPILIB], [-I\${includedir}])

ACX_LIB_NEW_LIB([ATHA1], [-L\${libdir} -latha1], [atha1])
ACX_LIB_ADD_PUBLIC_FLAGS([ATHA1], [-I\${includedir}])

ACX_LIB_NEW_PRG([XKAAPIPROGS])
ACX_LIB_NEW_PRG([ATHA1PROGS])

# add warning flags to the C and C++ compilers
AX_CFLAGS_WARN_ALL([AM_CFLAGS])
AX_CXXFLAGS_WARN_ALL([AM_CXXFLAGS])

AC_ARG_ENABLE([target],[AS_HELP_STRING([--enable-target=<target>], [mt[default],...])],[],
  [enable_target=default]
)
AS_CASE([$enable_target],
  [mt|default], [dnl
    AC_MSG_NOTICE([Using target MT])
    KAAPI_TARGET=mt
  ],[dnl
    AC_MSG_ERROR([Unknown target '$enable_target'. Aborting.])
  ])
AC_SUBST([KAAPI_TARGET])
AM_CONDITIONAL([KAAPI_TARGET_MT],[test x"$KAAPI_TARGET" = x"mt"])

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

AC_CHECK_FUNCS([pthread_attr_setaffinity_np],
  [AC_DEFINE([KAAPI_USE_SCHED_AFFINITY],1,[Use affinity interface])])

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

dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------
dnl --with && --enable OPTIONS
dnl -----------------------------------------------------------------------
dnl -----------------------------------------------------------------------

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

ACX_LIB_LINK([ATHA1], [PUBLIC], [XKAAPILIB])

ACX_LIB_LINK([XKAAPIPROGS], [XKAAPILIB])
ACX_LIB_LINK([ATHA1PROGS], [ATHA1])

AC_SUBST(AM_CPPFLAGS)
AC_SUBST(AM_CFLAGS)
AC_SUBST(AM_CXXFLAGS)

AM_LDFLAGS="$LDFLAGS"
AC_SUBST(AM_LDFLAGS)

AC_CONFIG_FILES([
	Makefile
	api-atha/Makefile
	examples/Makefile
	src/Makefile

	xkaapi.pc
	atha1.pc
	src/misc/kaapi_init.c
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
for flag in CFLAGS CPPFLAGS CXXFLAGS; do
  for f in AM@&t@_$flag $flag ; do
    val=`eval echo '"$'$f'"'`
    val=`echo "$val" | sed 's/^ *//'`
    if test x"$val" != x ; then
       list="$list
  $f: $val"
    fi
  done
done
AC_MSG_NOTICE([Default flags used:$list])
AC_MSG_NOTICE([*************************************])

