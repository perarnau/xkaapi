#
# SYNOPSIS
#
#   ACX_PROG_CCACHE
#   ACX_PROG_CCACHE_CC
#   ACX_PROG_CCACHE_CXX
#
# DESCRIPTION
#
#   These macro check for the 'ccache' program and prefix C and C++ compilers
#   with "ccache" if this later is present or provided
#
# LAST MODIFICATION
#
#   2010-11-13
#
# COPYLEFT
#
#   Copyright (c) 2008 Vincent Danjean <Vincent.Danjean@imag.fr>
#
#   This program is free software; you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation; either version 2 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#   General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
#   02111-1307, USA.
#
#   As a special exception, the respective Autoconf Macro's copyright
#   owner gives unlimited permission to copy, distribute and modify the
#   configure scripts that are the output of Autoconf when processing
#   the Macro. You need not follow the terms of the GNU General Public
#   License when using or distributing such scripts, even though
#   portions of the text of the Macro appear in them. The GNU General
#   Public License (GPL) does govern all other use of the material that
#   constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the
#   Autoconf Macro released by the Autoconf Macro Archive. When you
#   make and distribute a modified version of the Autoconf Macro, you
#   may extend this special exception to the GPL to apply to your
#   modified version as well.

AC_DEFUN([ACX_PROG_CCACHE], [
  AC_ARG_WITH(
    [ccache],
    [AS_HELP_STRING([--with-ccache=yes|no|check], [use ccache for compiling (default no)])],
    [ case "x$with_ccache" in
	xy*) with_ccache=yes ;;
	xn*) with_ccache=no ;;
	*) with_ccache=check ;;
      esac
    ],
    [with_ccache=no]
  )
  have_ccache=no
  AS_IF([ test $with_ccache != no ], [
    AC_PATH_PROGS([CCACHE],[m4_ifval([$1],[$1],[ccache])])
    AC_ARG_VAR([CCACHE],[ccache command])
    AS_IF([ test "x$CCACHE" = x ], [
      CCACHE="false ...ccache program not found..."
    ], [
      have_ccache=yes
    ])
  ])
])dnl ACX_PROG_CCACHE

AC_DEFUN([_ACX_PROG_CCACHE_C], [
  AC_REQUIRE([AC_PROG_$1])
  AC_REQUIRE([ACX_PROG_CCACHE])

  AS_VAR_PUSHDEF([FLAGS],[$2])dnl
  AS_VAR_PUSHDEF([VAR],[ac_cv_prog_ccache_$1])dnl
  AC_CACHE_CHECK([weather ccache is already invoked with '$$1'],
  VAR,[VAR="no"
   AC_LANG_PUSH([$3])
   ac_save_[]FLAGS="$[]FLAGS"
   acx_ccache_logfile="$(pwd)/config.ccache.log"
   rm -f "$acx_ccache_logfile"
   (
     CCACHE_LOGFILE="$acx_ccache_logfile"
     export CCACHE_LOGFILE
     AC_COMPILE_IFELSE(
	[AC_LANG_PROGRAM([],[[return 0;]])])
   )
   AS_IF([test -f "$acx_ccache_logfile"],
	[rm -f "$acx_ccache_logfile" ; VAR=yes])
   FLAGS="$ac_save_[]FLAGS"
   AC_LANG_POP([$3])
  ])
  case "$VAR,$have_ccache,$with_ccache" in
    yes,*) AC_MSG_NOTICE([ccache already enabled. Not doing anything.]) ;;
    no,no,yes) AC_MSG_ERROR([ccache not present. Cannot using it.]) ;;
    no,no,check) AC_MSG_NOTICE([ccache not present. Not using it.]) ;;
    no,no,no) AC_MSG_NOTICE([ccache not used]) ;;
    no,yes,*) AC_MSG_NOTICE([ccache present. Prefixing '$$1' with '$CCACHE'])
      $1="$CCACHE $$1"
      ;;
    *) AC_MSG_ERROR([Internal error, please report '$VAR,$have_ccache,$with_ccache']) ;;
  esac
  AS_VAR_POPDEF([VAR])dnl
  AS_VAR_POPDEF([FLAGS])dnl

])dnl _ACX_PROG_CCACHE_C

AC_DEFUN([ACX_PROG_CCACHE_CC], [
  _ACX_PROG_CCACHE_C([CC], [CFLAGS], [C])
])

AC_DEFUN([ACX_PROG_CCACHE_CXX], [
  _ACX_PROG_CCACHE_C([CXX], [CXXFLAGS], [C++])
])


