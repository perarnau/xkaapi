# SYNOPSIS
#
#   AX_PROG_ESED
#
# DESCRIPTION
#
#   Check how to use Extended Regular Expression with sed. It defines the
#   variable ESED for further use, which you should in fact
#   treat like it used to be with be SED. 
#   The actual value is assured to be the value of SED with any required
#   options (or empty if no sed and/or options have been found).
#
# LICENSE
#
#   Copyright (c) 2011 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 3 of the License, or (at your
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

#   Version 20110118

AU_ALIAS([AC_PROG_ESED], [AX_PROG_ESED])
AC_DEFUN([AX_PROG_ESED],
[AC_REQUIRE([AC_PROG_SED])dnl
AC_MSG_CHECKING(whether sed accepts ERE)
AC_CACHE_VAL(ac_cv_prog_ESED,
[rm -f conftestdata
if test x"$SED" = x ; then # no sed available
  ac_cv_prog_ESED=""
elif test xb != x`echo a | $SED -e 's/a/b/'`; then # no working sed
  ac_cv_prog_ESED=""
elif test xb = x`echo aa | $SED -e 's/a+/b/'`; then
  ac_cv_prog_ESED="$SED"
elif test xb = x`echo aa | $SED -r -e 's/a+/b/'`; then
  ac_cv_prog_ESED="$SED -r"
elif test xb = x`echo aa | $SED -E -e 's/a+/b/'`; then
  ac_cv_prog_ESED="$SED -E"
else
  ac_cv_prog_ESED=""
fi])dnl
ESED="$ac_cv_prog_ESED"
if test x"$ESED" = "x"; then
  AC_MSG_RESULT(no)
else
  AC_MSG_RESULT([yes, using $ESED])
fi
AC_SUBST(ESED)dnl
])
