#
# SYNOPSIS
#
#   AMX_SILENT_RULES([default], [action-if-exists], [action-if-not-exists])
#
# DESCRIPTION
#
#   These macro calls AM_SILENT_RULES and set AM_DEFAULT_VERBOSITY
#   if the AM_SILENT_RULES exists (ie automake >= 1.11) 
#
# LAST MODIFICATION
#
#   2010-01-13
#
# COPYLEFT
#
#   Copyright (c) 2010 Vincent Danjean <Vincent.Danjean@imag.fr>
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

AC_DEFUN([AMX_SILENT_RULES], [
  m4_ifdef([AM_SILENT_RULES], [
    AS_CASE([$1],
      [yes|no], [dnl
        if test "${enable_silent_rules+set}" != set; then :
          # choose default, must be yes or no (or empty)
          enable_silent_rules=$1;
        fi
      ], [""], [:], [dnl
        AC_MSG_ERROR([Invalid default '$1' for [AMX_SILENT_RULES]])
      ])
    AM_SILENT_RULES
    $2
  ], [
    AC_MSG_NOTICE([Not using silent Makefile rules as automake 1.11 is required for this])
    $3
  ]) 
])dnl AMX_SILENT_RULES
