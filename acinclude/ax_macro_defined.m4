# SYNOPSIS
#
#   AX_MACRO_DEFINED (macro [,includes, [action-if-defined, [action-if-not-defined]]])
#
# DESCRIPTION
#
#   Check if the macro 'macro' is defined
#
#    - $1 CPP macro to check if defined
#    - $2 includes to add to define the macro
#    - $3 action-if-defined : nothing
#    - $4 action-if-not-defined : nothing
#
# LICENSE
#
#   Copyright (c) 2010 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_MACRO_DEFINED],[dnl
AS_VAR_PUSHDEF([FLAGS],[CFLAGS])dnl
AS_VAR_PUSHDEF([VAR],[ac_cv_macro_defined_]AS_TR_SH([$1]))dnl
AC_CACHE_CHECK([whether C compiler defines $1],
VAR,[VAR="no"
 AC_LANG_PUSH([C])
 AC_COMPILE_IFELSE(
     [AC_LANG_PROGRAM([$2
#if !defined($1)
#error $1 not defined
#endif
],[[return 0;]])],
     [VAR="yes"])
 AC_LANG_POP([C])
])
if test "$VAR" = yes ; then :
	$3
else :
	$4
fi
AS_VAR_POPDEF([VAR])dnl
])

dnl the only difference - the LANG selection... and the default FLAGS

AC_DEFUN([AX_CXXFLAGS_WARN_ALL],[dnl
AS_VAR_PUSHDEF([FLAGS],[CXXFLAGS])dnl
AS_VAR_PUSHDEF([VAR],[ac_cv_cxxflags_warn_all])dnl
AC_CACHE_CHECK([m4_ifval($1,$1,FLAGS) for maximum warnings],
VAR,[VAR="no, unknown"
 AC_LANG_PUSH([C++])
 ac_save_[]FLAGS="$[]FLAGS"
for ac_arg dnl
in "-pedantic  % -Wall"       dnl   GCC
   "-xstrconst % -v"          dnl Solaris C
   "-std1      % -verbose -w0 -warnprotos" dnl Digital Unix
   "-qlanglvl=ansi % -qsrcmsg -qinfo=all:noppt:noppc:noobs:nocnd" dnl AIX
   "-ansi -ansiE % -fullwarn" dnl IRIX
   "+ESlit     % +w1"         dnl HP-UX C
   "-Xc        % -pvctl[,]fullmsg" dnl NEC SX-5 (Super-UX 10)
   "-h conform % -h msglevel 2" dnl Cray C (Unicos)
   #
do FLAGS="$ac_save_[]FLAGS "`echo $ac_arg | sed -e 's,%%.*,,' -e 's,%,,'`
   AC_COMPILE_IFELSE(
     [AC_LANG_PROGRAM([],[[return 0;]])],
     [VAR=`echo $ac_arg | sed -e 's,.*% *,,'` ; break])
done
 FLAGS="$ac_save_[]FLAGS"
 AC_LANG_POP([C++])
])
case ".$VAR" in
     .ok|.ok,*) m4_ifvaln($3,$3) ;;
   .|.no|.no,*) m4_ifvaln($4,$4,[m4_ifval($2,[
        AC_RUN_LOG([: m4_ifval($1,$1,FLAGS)="$m4_ifval($1,$1,FLAGS) $2"])
                      m4_ifval($1,$1,FLAGS)="$m4_ifval($1,$1,FLAGS) $2"])]) ;;
   *) m4_ifvaln($3,$3,[
   if echo " $[]m4_ifval($1,$1,FLAGS) " | grep " $VAR " 2>&1 >/dev/null
   then AC_RUN_LOG([: m4_ifval($1,$1,FLAGS) does contain $VAR])
   else AC_RUN_LOG([: m4_ifval($1,$1,FLAGS)="$m4_ifval($1,$1,FLAGS) $VAR"])
                      m4_ifval($1,$1,FLAGS)="$m4_ifval($1,$1,FLAGS) $VAR"
   fi ]) ;;
esac
AS_VAR_POPDEF([VAR])dnl
AS_VAR_POPDEF([FLAGS])dnl
])

dnl  implementation tactics:
dnl   the for-argument contains a list of options. The first part of
dnl   these does only exist to detect the compiler - usually it is
dnl   a global option to enable -ansi or -extrawarnings. All other
dnl   compilers will fail about it. That was needed since a lot of
dnl   compilers will give false positives for some option-syntax
dnl   like -Woption or -Xoption as they think of it is a pass-through
dnl   to later compile stages or something. The "%" is used as a
dnl   delimimiter. A non-option comment can be given after "%%" marks
dnl   which will be shown but not added to the respective C/CXXFLAGS.
