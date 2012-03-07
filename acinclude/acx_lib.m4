dnl acx_lib.m4 - Macros to manage flags/libs when creating libraries.   -*- Autoconf -*-
dnl
dnl Copyright © 2008,2009 Vincent Danjean <vdanjean@debian.org>
dnl
dnl     Copying and distribution of this file, with or without modification,
dnl     are permitted in any medium without royalty provided the copyright
dnl     notice and this notice are preserved.
dnl
dnl @author Vincent Danjean <Vincent.Danjean@imag.fr>
dnl @version 2008-06-15
dnl @license AllPermissive
dnl @category InstalledPackages
dnl
dnl ==================================================================
dnl These macros aims to help to manage library dependencies.
dnl When a new library needs another library, the developer needs/wants
dnl to manage:
dnl * flags/libs needed to compile (and link in case of shared library)
dnl   its files
dnl * flags, libs and packages, public and private, needed in the
dnl   library.pc file (to be used by other programs with pkg-config
dnl   to use the new library)
dnl * flags and libs needed in the library-config script (to be used
dnl   by other programs to use the new library)
dnl Macros in this file help the developer to manage all these flags.
dnl 
dnl These macros can be separated in several groups:
dnl * Macros to help to get the various flags/libs for a specific
dnl   external library
dnl   + ACX_LIB_CHECK_PLAIN: get infos with user macros/tests
dnl   + ACX_LIB_CHECK_CONFIG: get infos with "library-config" like scripts
dnl   + ACX_LIB_CHECK_PKGCONFIG: get infos with pkg-config
dnl   + ACX_LIB_NOLIB: the corresponding external library will not be used
dnl * Macros to manage flags/libs for a new library
dnl   + ACX_LIB_NEW_LIB: declare a new library
dnl   + ACX_LIB_NEW_PRG: declare a new program
dnl   + ACX_LIB_LINK: add some library dependencies to a new library
dnl   Dependencies can be (see below for an example)
dnl   - private (no header nor symbol are exposed by the new library)
dnl   - public (headers and symbols are exposed by the new library)
dnl   - half-private (headers but no symbols are exposed by the new library)
dnl * Macros to add new flags
dnl   + ACX_LIB_ADD_BUILD_FLAGS: new libraries will use these flags for their
dnl     build
dnl   + ACX_LIB_ADD_PUBLIC_FLAGS: new libraries will have these flags in
dnl     their public config flag
dnl   Note: if applied to a external library, these macros define flags that
dnl   will be used with new libraries linked to this external one
dnl * Various helper macros
dnl   + ACX_LIB_SHOW_LIB: display information about an external or
dnl     a new library
dnl   + ACX_LIB_SHOW_EXTERNAL_LIBS: display information about all
dnl     external libraries
dnl   + ACX_LIB_SHOW_NEW_LIBS: display information about all new
dnl     libraries
dnl   + ACX_LIB_SHOW_NEW_PRGS: display information about all new
dnl     programs
dnl   + ACX_LIB_SHOW: display information about all (external and
dnl     new) libraries and all programs
dnl   + ACX_LIB_CLEANUP_NEW_OBJ: remove duplicate flags or libraries in the
dnl     variables of a new library or program
dnl   + ACX_LIB_CLEANUP_NEW_LIBS: remove duplicate flags or libraries in the
dnl     variables of all new libraries
dnl   + ACX_LIB_CLEANUP_NEW_PRGS: remove duplicate flags or libraries in the
dnl     variables of all new programs
dnl   + ACX_LIB_CLEANUP: remove duplicate flags or libraries in the
dnl     variables of all new libraries and programs
dnl
dnl
dnl   Here is a quick example of the three dependency situations
dnl   A is a new library, B is a library used by A, P is a program using A
dnl
dnl   Common part:
dnl   ============
dnl   P.c: #include "A.h"
dnl        int main() { ... ; A_t ret=fn_A(); ... }
dnl   B-type.h: typedef struct {...} B_t;
dnl   B.h: #include "B-type.h"
dnl        B_t fn_B(int);
dnl   B.c: #include "B.h"
dnl        B_t fn_B(int var) {...}
dnl   Compile command for shared objects:
dnl   A.o:   gcc -c A.c -I/inc_B
dnl   P.o:   gcc -c P.c $CFLAGS
dnl   Link command for shared objects:
dnl   libA:  gcc -shared A.o  -o libA.so -lB
dnl   P:     gcc -shared P.o $LIBS
dnl
dnl   Specific part:
dnl   ==============
dnl   Model |     Private      |      Public          |   half-private
dnl         |                  |                      |
dnl   A.h:  |typedef struct {  |#include "B.h"        |#include "B-type.h"
dnl         |  int var;        |typedef B_t A_t;      |typedef struct {
dnl         |} A_t             |#define fn_A() fn_B(0)|  B_t var;
dnl         |A_t fn_A();       |A_t fn_A2();          |} A_t;
dnl         |                  |                      |A_t fn_A();
dnl         |                  |                      |
dnl   A.c:  |#include "A.h"    |#include "A.h"        |#include "A.h"
dnl         |#include "B.h"    |                      |#include "B.h"
dnl         |A_t fn_A() {      |A_t fn_A2() {         |A_t fn_A() {
dnl         |  B_t foo=fn_B(0);|  B_t foo=fn_B(0);    |  B_t foo=fn_B(0);
dnl         |  ...             |  ...                 |  A_t res; res.var=foo;
dnl         |}                 |}                     |  ...
dnl         |                  |                      |}
dnl         |                  |                      |
dnl   LIBS  |-lA               |-lA -lB               |-lA
dnl   CFLAGS|-I/inc_A          |-I/inc_A -I/inc_B     |-I/inc_A -I/inc_B
dnl
dnl  Of course, the same analyse needs to be done for static linking...
dnl
dnl ================================================================
dnl ================================================================
dnl PUBLIC MACROS DOCUMENTATION
dnl ================================================================
dnl ================================================================
dnl
dnl ================================================================
dnl * Macros to help to get the various flags/libs for a specific
dnl   external library
dnl ================================================================
dnl ACX_LIB_CHECK_*(VARIABLE-PREFIX, [prefix], ...)
dnl ACX_LIB_NOLIB
dnl 
dnl All these macros, will set the variable 'acx_lib_have_NAME' to
dnl "yes" or "no" depending whether the library if found or not
dnl (always "no" for ACX_LIB_NOLIB)
dnl
dnl The following output variables are set with AC_SUBST (where NAME
dnl is VARIABLE-PREFIX):
dnl   NAME_VERSION        Library version (or "unknown")
dnl   NAME_CPPFLAGS       Preprocessor compiler flags
dnl   NAME_LIBS           Shared libraries
dnl   NAME_LIBS_STATIC    Static (and shared) libraries
dnl   NAME_CFLAGS         C compiler flags
dnl   NAME_CXXFLAGS       C++ compiler flags
dnl   NAME_LDFLAGS        Library flags
dnl   NAME_LDFLAGS_STATIC Library flags for static link
dnl   NAME_PKGCONFIG      pkg-config depends
dnl
dnl Some cleanup is automatically done at the end of these macros just before
dnl executing ACTION-IF-FOUND:
dnl - if NAME_LIBS_STATIC is not set where as NAME_LIBS is,
dnl   NAME_LIBS_STATIC is set to NAME_LIBS
dnl - if NAME_VERSION is empty, NAME_VERSION is set to "unknown"
dnl - if NAME_CFLAGS or NAME_CXXFLAGS are set and NAME_CPPFLAGS is not
dnl   a warning is emitted. Setting NAME_CPPFLAGS (even with an empty
dnl   value) makes the warning disapparaired (a classical error is
dnl   to use CFLAGS instead of CPPFLAGS)
dnl
dnl If 'prefix' is set to something else than "yes", "no" or "default",
dnl it is used as a location prefix for the library.
dnl
dnl
dnl          /-------------------\
dnl @synopsis ACX_LIB_CHECK_PLAIN(VARIABLE-PREFIX, [prefix], TESTS,
dnl [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl 
dnl This macro checks for the existence of a library with manual tests
dnl in the TESTS argument. These tests must call one of these two macros:
dnl ACX_LIB_NOTFOUND
dnl   in case the library is not found
dnl ACX_LIB_FOUND(ID="value", ...)
dnl   in case the library is found. This macro takes a variable number
dnl   of arguments where ID (case insensitive) can be one of the
dnl   following: Version, CppFlags, CFlags, CxxFlags, Libs, Libs-Static,
dnl   LDFlags, LDFlags-static and PkgConfig.
dnl   The various NAME_* variables will be set with these values
dnl   (default rules for NAME_* variables still apply)
dnl
dnl The globals variables CFLAGS, CPPFLAGS, CXXFLAGS and LIBS variables
dnl are saved before and restored and after the tests.
dnl
dnl During TESTS, ACX_LIB_CPPFLAGS and ACX_LIB_LIBS variables are set to
dnl "-Iprefix" and "-Lprefix" if there is a prefix.
dnl
dnl          /--------------------\
dnl @synopsis ACX_LIB_CHECK_CONFIG(VARIABLE-PREFIX, [prefix], library-config,
dnl ID="options", ...,
dnl [STOP, [TESTS = ACX_LIB_FOUND], [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND]])
dnl 
dnl This macro checks for the existence of a library with a library-config
dnl like program. Arguments ID="options" are arguments to pass to
dnl the library-config program to get the requested informations. The
dnl following ID (case insensitive) can be used:
dnl - Modules: argument added at the end of all invocation of library-config
dnl   [ie: library-config options-for-action modules]
dnl - Version: options to get library version
dnl - CppFlags: options to get preprocessor flags
dnl - Libs: options to get libraries to be linked with
dnl - Libs-Static: options to get all libraries to be linked with
dnl   in case of static link
dnl - CFlags: options to get C compiler flags
dnl - CxxFlags: options to get C++ compiler flags
dnl - LDFlags: options to get linker flags
dnl - LDFlags-Static: options to get static linker flags
dnl
dnl If any of the following parameters are used, a 'STOP' argument must
dnl be passed to indicate the end of the ID="option" arguments
dnl 
dnl - TESTS: code to ckeck that the correct library is here (can be used to
dnl   check the minimal version, ...)
dnl   When executed, some variables are set:
dnl   * NAME_CONFIG is set to the script path
dnl   * NAME_VERSION is set with the output of the config script (if
dnl     Version has been set)
dnl   TESTS must call one of these two macros:
dnl   * ACX_LIB_NOTFOUND: in case the library-config program cannot be used
dnl   * ACX_LIB_FOUND: in case the library-config program can be used
dnl     ACX_LIB_FOUND can have arguments (see ACX_LIB_CHECK_PLAIN) whose
dnl     values (if set) will be used instead of invoking 'library-config'.
dnl     For example, you can set a value for PkgConfig...
dnl
dnl If there is a prefix, library-config is searched in prefix/bin
dnl
dnl The NAME_CONFIG variable is set with AC_SUBST and can be override in the
dnl environment.
dnl
dnl          /-----------------------\
dnl @synopsis ACX_LIB_CHECK_PKGCONFIG(VARIABLE-PREFIX, [prefix], MODULES,
dnl [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl
dnl This macro checks for the existence of a library with the help of
dnl the pkg-config program.
dnl 
dnl If the pkg-config module is present, NAME_PKGCONFIG is set to MODULES.
dnl NAME_CFLAGS, NAME_LIBS and NAME_LIBS_STATIC can be set to override
dnl auto-detection.
dnl
dnl If there is a prefix, pc files will be searched in prefix/lib/pkgconfig
dnl
dnl The PKG_CONFIG variable is set with AC_SUBST and can be override in the
dnl environment.
dnl
dnl          /------------\
dnl @synopsis ACX_LIB_NOLIB(VARIABLE-PREFIX)
dnl
dnl 'acx_lib_have_NAME' is set to "no"
dnl In addition, the various 'NAME_*' variables are set to ""
dnl
dnl ================================================================
dnl * Macros to manage flags/libs for a new library
dnl ================================================================
dnl
dnl          /--------------\
dnl @synopsis ACX_LIB_NEW_LIB(VARIABLE-PREFIX, [link-flags],
dnl [pkg-config-module])
dnl
dnl Declare a new library and optionally tells how to link with this library.
dnl A new library can be linked with another new library ONLY if 'link-flags'
dnl is set.
dnl
dnl The following output variables are set with AC_SUBST (where NAME
dnl is VARIABLE-PREFIX):
dnl   NAME_BUILD_CPPFLAGS
dnl   NAME_BUILD_CFLAGS
dnl   NAME_BUILD_CXXFLAGS
dnl   NAME_BUILD_LDFLAGS
dnl   NAME_BUILD_LDFLAGS_STATIC
dnl   NAME_BUILD_LIBS
dnl   NAME_BUILD_LIBS_STATIC
dnl   NAME_PKGCONFIG_REQUIRES
dnl   NAME_PKGCONFIG_REQUIRES_PRIVATE
dnl   NAME_PKGCONFIG_LIBS
dnl   NAME_PKGCONFIG_LIBS_PRIVATE
dnl   NAME_PKGCONFIG_CFLAGS
dnl   NAME_CONFIG_CPPFLAGS
dnl   NAME_CONFIG_CFLAGS
dnl   NAME_CONFIG_CXXFLAGS
dnl   NAME_CONFIG_LDFLAGS
dnl   NAME_CONFIG_LDFLAGS_STATIC
dnl   NAME_CONFIG_LIBS
dnl   NAME_CONFIG_LIBS_STATIC
dnl
dnl All these variables are computed from external library variables
dnl when added as dependecy to the new library (see ACX_LIB_LINK below).
dnl NAME_BUILD_* variables are expected to be used in the build system
dnl   (ie in Makefile.am).
dnl NAME_PKGCONFIG_* variables are expected to be used in a name.pc file.
dnl NAME_CONFIG_* variables are expected to be used in a name-config script.
dnl
dnl Note: new library will NEVER appaired in the NAME_BUILD_* variables.
dnl Indeed, Makefile.am will prefer setup a relative path to the libname.la
dnl built file.
dnl
dnl          /--------------\
dnl @synopsis ACX_LIB_NEW_PRG(VARIABLE-PREFIX, [LIBS...])
dnl
dnl Declare a new program. It is similar to ACX_LIB_NEW_LIB but only
dnl NAME_BUILD_* variables are AC_SUBST-ed.
dnl A list a library to be linked with can be given immediately (instead
dnl or in addition to call ACX_LIB_LINK after)
dnl
dnl          /----------------\
dnl @synopsis ACX_LIB_LINK(VARIABLE-PREFIX, MODE1, LIBS... [, MODE2, LIBS..., ...])
dnl @synopsis ACX_LIB_LINK(VARIABLE-PREFIX, LIBS...)
dnl
dnl Add some external libraries as dependencies to an already declared new
dnl library or new program. The various variables are updated accordingly.
dnl
dnl MODE can be PRIVATE, PUBLIC, and HALF-PRIVATE (case independent)
dnl
dnl Note: without MODE, private linkage is assumed
dnl Note: is case of new program, only private libraries are accepted.
dnl
dnl ================================================================
dnl * Macros to add new flags
dnl ================================================================
dnl These macros, if applied to a external library, define flags that
dnl will be used with all new libraries linked to the external library.
dnl
dnl          /-----------------------\
dnl @synopsis ACX_LIB_ADD_BUILD_FLAGS(VARIABLE-PREFIX, [CPPFLAGS], [CFLAGS],
dnl   [CXXFLAGS], [LDFLAGS], [LDFLAGS_STATIC])
dnl
dnl The flags will be added to the corresponding NAME_BUILD_* variables.
dnl
dnl Note: to avoid too long compilation line, it is better to use AC_DEFINE
dnl than to call ACX_LIB_ADD_BUILD_FLAGS(NAME, [-Dsymbol])
dnl
dnl          /-----------------------\
dnl @synopsis ACX_LIB_ADD_PUBLIC_FLAGS(VARIABLE-PREFIX, [CPPFLAGS], [CFLAGS],
dnl   [CXXFLAGS], [LDFLAGS], [LDFLAGS_STATIC])
dnl
dnl The flags will be added to the corresponding NAME_[PKG]?CONFIG_* variables.
dnl
dnl These flags will be added even if the library is not found.
dnl Typical use is in ACTION-IF-FOUND orACTION-IF-NOFOUND  blocks:
dnl ACX_LIB_ADD_PUBLIC_FLAGS(NAME, "-DUSE_NAME") or
dnl ACX_LIB_ADD_PUBLIC_FLAGS(NAME, "-DDONT_USE_NAME")
dnl 
dnl Note: these flags will NOT be added in the NAME_BUILD_* variables.
dnl
dnl ================================================================
dnl * Various helper macros
dnl ================================================================
dnl          /----------------\
dnl @synopsis ACX_LIB_SHOW_LIB(VARIABLE-PREFIX)
dnl
dnl This macro displays information about an external or a new library.
dnl Only non-null variables are shown
dnl
dnl          /--------------------------\
dnl @synopsis ACX_LIB_SHOW_EXTERNAL_LIBS
dnl
dnl This macro displays information about all external libraries.
dnl
dnl          /---------------------\
dnl @synopsis ACX_LIB_SHOW_NEW_LIBS
dnl
dnl This macro displays information about all new libraries.
dnl
dnl          /------------\
dnl @synopsis ACX_LIB_SHOW
dnl
dnl This macro displays information about all external and new libraries.
dnl
dnl 
dnl          /-------------------\
dnl @synopsis ACX_LIB_CLEANUP_OBJ(VARIABLE-PREFIX, [noflags nolibs])
dnl
dnl This macros remove all duplicate flags and libraries in the various
dnl variables of new libraries or new programs:
dnl * for duplicated flags, only the first one is kept
dnl * flags are removed from X_CFLAGS if they are already in X_CPPFLAGS
dnl * flags are removed from X_CXXFLAGS if they are already in X_CPPFLAGS
dnl * for duplicated libraries (-lname), only the last one is kept
dnl * for each X_LIBS_STATIC, a new AC_SUBST variable X_LIBS_STATIC_ONLY
dnl   is created and X_LIBS_STATIC="$X_LIBS $X_LIBS_STATIC_ONLY"
dnl   X_LIBS_STATIC and X_LIBS_STATIC_ONLY does not have any duplicates
dnl
dnl noflags or nolibs can be used to restrict the operations.
dnl
dnl Note: this macros split flags and libs with whitespaces, so do not call
dnl it if some of your flags have spaces in them (-DFOO="my name")
dnl
dnl Example
dnl Before:
dnl   NAME_BUILD_CPPFLAGS= -DFOO -DBAR -DFOO
dnl   NAME_BUILD_CFLAGS= -DBAR -DLANG=C
dnl   NAME_BUILD_LIBS= -L/lib/foo -lfoo -lbar -pthread -L/lib/foo -lfoo
dnl   NAME_BUILD_LIBS_STATIC= -L/lib/foo -lbaz -lbar -L/lib/bar
dnl After:
dnl   NAME_BUILD_CPPFLAGS= -DFOO -DBAR -DFOO
dnl   NAME_BUILD_CFLAGS= -DBAR -DLANG=C
dnl   NAME_BUILD_LIBS= -L/lib/foo -lbar -pthread -lfoo
dnl   NAME_BUILD_LIBS_STATIC= -L/lib/foo -lbar -pthread -lfoo -lbaz -L/lib/bar
dnl   NAME_BUILD_LIBS_STATIC_ONLY= -lbaz -L/lib/bar
dnl
dnl          /------------------------\
dnl @synopsis ACX_LIB_CLEANUP_NEW_LIBS([noflags nolibs])
dnl
dnl Apply ACX_LIB_CLEANUP_OBJ to all new libraries
dnl
dnl          /------------------------\
dnl @synopsis ACX_LIB_CLEANUP_NEW_PRGS([noflags nolibs])
dnl
dnl Apply ACX_LIB_CLEANUP_OBJ to all new programs
dnl
dnl          /---------------\
dnl @synopsis ACX_LIB_CLEANUP([noflags nolibs])
dnl
dnl Apply ACX_LIB_CLEANUP_OBJ to all new libraries and new programs
dnl
dnl
dnl ================================================================
dnl ================================================================
dnl IMPLEMENTATION
dnl ================================================================
dnl ================================================================
dnl
dnl Macros starting with ACX_LIB_* are public interface.
dnl They should remain stable and fully documented.
dnl
dnl Macros starting with _ACX_LIB_* are private interface and can be
dnl changed without any advertisement.
dnl
dnl Unless stated otherwise in documentation, all variables are
dnl private interface (ie m4 programmer must not rely on them).
dnl
dnl
dnl ====================================================================
dnl ====================================================================
dnl Internal macros for initialisation and static checks
dnl ----------------------------------------------------
dnl 
dnl Internal initialisation. Required by all public macros
AC_DEFUN([_ACX_LIB_INIT],
[m4_pattern_forbid([^_?ACX_LIB_[A-Z_]+$])dnl
m4_pattern_allow([^ACX_LIB_(CPPFLAGS|LIBS)$])dnl
m4_define([_ACX_LIB_LIST_EXTLIBS],[])
m4_define([_ACX_LIB_LIST_NEWPRGS],[])
m4_define([_ACX_LIB_LIST_NEWLIBS],[])
m4_define([_ACX_LIB_LIST_NEWEXTLIBS],[])
])dnl _ACX_LIB_INIT
dnl
dnl 
AC_DEFUN([_ACX_LIB_ENFORCE_NAME],
[m4_bmatch([$1],[^[A-Za-z_][A-Za-z0-9_]*$],
   [],
   [m4_fatal([Invalid name $1 (must be [A-Za-z_][A-Za-z_]*)])]
   )dnl
])dnl _ACX_LIB_INIT
dnl
dnl
AC_DEFUN([_ACX_LIB_VAR_EXTLIBS],
[VERSION CPPFLAGS LIBS LIBS_STATIC CFLAGS CXXFLAGS dnl
LDFLAGS LDFLAGS_STATIC PKGCONFIG])
dnl
dnl
AC_DEFUN([_ACX_LIB_VAR_NEWPRGS],
[INFO dnl
BUILD_CPPFLAGS BUILD_CFLAGS BUILD_CXXFLAGS dnl
BUILD_LDFLAGS BUILD_LDFLAGS_STATIC BUILD_LIBS BUILD_LIBS_STATIC dnl
])
dnl
dnl
AC_DEFUN([_ACX_LIB_VAR_NEWLIBS_ONLY],
[dnl
PKGCONFIG_REQUIRES PKGCONFIG_REQUIRES_PRIVATE dnl
PKGCONFIG_LIBS PKGCONFIG_LIBS_PRIVATE PKGCONFIG_CFLAGS dnl
CONFIG_CPPFLAGS CONFIG_CFLAGS CONFIG_CXXFLAGS dnl
CONFIG_LDFLAGS CONFIG_LDFLAGS_STATIC dnl
CONFIG_LIBS CONFIG_LIBS_STATIC dnl
])
dnl
dnl
AC_DEFUN([_ACX_LIB_VAR_NEWLIBS],
[_ACX_LIB_VAR_NEWPRGS dnl
_ACX_LIB_VAR_NEWLIBS_ONLY dnl
])
dnl
dnl m4_argn is not available in autoconf...
AC_DEFUN([_ACX_LIB_argn], [m4_if([$1], 1, [[$2]],
       [_ACX_LIB_argn(m4_decr([$1]), m4_shift(m4_shift($@)))])])
dnl 
dnl @synopsis _ACX_LIB_IS_NEWEXTLIB(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_NEWEXTLIB],
[m4_if(m4_index([ ]_ACX_LIB_LIST_NEWEXTLIBS[ ],[ $1 ]),[-1],[$3],[$2])])
dnl 
dnl @synopsis _ACX_LIB_IS_NEWLIB(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_NEWLIB],
[m4_if(m4_index([ ]_ACX_LIB_LIST_NEWLIBS[ ],[ $1 ]),[-1],[$3],[$2])])
dnl 
dnl @synopsis _ACX_LIB_IS_NEWPRG(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_NEWPRG],
[m4_if(m4_index([ ]_ACX_LIB_LIST_NEWPRGS[ ],[ $1 ]),[-1],[$3],[$2])])
dnl 
dnl@synopsis _ACX_LIB_IS_EXTLIB(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_EXTLIB],
[m4_if(m4_index([ ]_ACX_LIB_LIST_EXTLIBS[ ],[ $1 ]),[-1],[$3],[$2])])
dnl 
dnl@synopsis _ACX_LIB_IS_LIB(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_LIB],
[_ACX_LIB_IS_NEWLIB([$1],[$2],[_ACX_LIB_IS_EXTLIB([$1],[$2],[$3])])])
dnl 
dnl@synopsis _ACX_LIB_IS_NEWOBJ(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_NEWOBJ],
[_ACX_LIB_IS_NEWLIB([$1],[$2],[_ACX_LIB_IS_NEWPRG([$1],[$2],[$3])])])
dnl 
dnl@synopsis _ACX_LIB_IS_DEFINED(VAR-PREFIX,IF-TRUE,IF-FALSE)
AC_DEFUN([_ACX_LIB_IS_DEFINED],
[_ACX_LIB_IS_LIB([$1],[$2],[_ACX_LIB_IS_NEWPRG([$1],[$2],[$3])])])
dnl 
dnl@synopsis _ACX_LIB_NAME(VAR-PREFIX)
AC_DEFUN([_ACX_LIB_NAME],
[_ACX_LIB_IS_NEWPRG([$1],[new program],
  [_ACX_LIB_IS_NEWLIB([$1],[new library],
    [_ACX_LIB_IS_EXTLIB([$1],[external library],
      [undefined object])])])])
dnl ====================================================================
dnl ====================================================================
dnl Internal macros to manage shell variables
dnl -----------------------------------------
dnl
dnl @synopsis _ACX_LIB_VAR_ADD(VAR-PREFIX,VAR-SUFFIX,VALUE,[SEPARATOR])
AC_DEFUN([_ACX_LIB_VAR_ADD],
[test x"$3" != x && $1[]_$2="$[]{$1[]_$2[]:+${$1[]_$2}m4_ifval([$4], [$4], [ ])}$3"])
])dnl _ACX_LIB_VAR_ADD
dnl
dnl @synopsis _ACX_LIB_VAR_PUSH(VAR,[value])
AC_DEFUN([_ACX_LIB_VAR_PUSH],
[dnl
  _acx_lib_var_[]$1[]_set=${$1+set}
  _acx_lib_var_[]$1[]_value="${$1}"
  m4_ifval([$2],[$1="$2"])
])dnl _ACX_LIB_PUSH_VAR
dnl
dnl @synopsis _ACX_LIB_VAR_POP(VAR)
AC_DEFUN([_ACX_LIB_VAR_POP],
[AS_IF([test x"${_acx_lib_var_[]$1[]_set+set}" != xset],
	     [AC_MSG_ERROR(m4_location[: Internal error: variable $1 has not been saved])],
       [test x"${_acx_lib_var_[]$1[]_set}" = xset],
             [$1="${_acx_lib_var_[]$1[]_value}"],
       [unset "$1"])dnl
])dnl _ACX_LIB_POP_VAR
dnl
dnl @synopsis _ACX_LIB_VAR_PUSH_EXTLIB(EXTLIB)
AC_DEFUN([_ACX_LIB_VAR_PUSH_EXTLIB],
[_ACX_LIB_IS_EXTLIB([$1],[],[m4_fatal([$1 is not an external library])])dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
    _ACX_LIB_VAR_PUSH([$2[]_[]_ACX_LIB_SUFFIX])
    m4_bmatch(_ACX_LIB_SUFFIX,
      [VERSION],[$2[]_VERSION="Not used"],
      [PKGCONFIG], [$2[]_[]_ACX_LIB_SUFFIX=""],
      [FLAGS], [$2[]_[]_ACX_LIB_SUFFIX=""],
      [LIBS], [$2[]_[]_ACX_LIB_SUFFIX=""],
      [m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_VAR_PUSH_EXTLIB])]
    )
  ])dnl      	  
])
dnl
dnl @synopsis _ACX_LIB_VAR_POP_EXTLIB(EXTLIB)
AC_DEFUN([_ACX_LIB_VAR_POP_EXTLIB],
[_ACX_LIB_IS_EXTLIB([$1],[],[m4_fatal([$1 is not an external library])])dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
    _ACX_LIB_VAR_POP([$2[]_[]_ACX_LIB_SUFFIX])
  ])dnl      	  
])
dnl ====================================================================
dnl ====================================================================
dnl Macros to display information
dnl -----------------------------
dnl
dnl
AC_DEFUN([_ACX_LIB_SHOW_ALL], [_ACX_LIB_LIST_EXTLIBS _ACX_LIB_LIST_NEWLIBS])
dnl
dnl @synopsis _ACX_LIB_SHOW_NOTICE(LIB,STR-PREFIX,VARNAME,VARLIB-SUFFIX,[force],[HIDE-VALUE])
AC_DEFUN([_ACX_LIB_SHOW_NOTICE],
[dnl
  acx_lib_show_value="$[]$1[]_[]$4"
  if test x$5"$acx_lib_show_value" != x m4_ifval([$6],dnl
      [ && test x"$acx_lib_show_value" != x"$6" ]); then
    acx_lib_show_txt="$acx_lib_show_txt
AS_HELP_STRING($2[]$3[:],[$acx_lib_show_value])"
  fi
])dnl _ACX_LIB_SHOW_NOTICE
dnl
dnl @synopsis _ACX_LIB_SHOW_NOTICE_FLAGS(LIB,STR-PREFIX,VARNAME,VARLIB-SUFFIX,force)
AC_DEFUN([_ACX_LIB_SHOW_NOTICE_FLAGS],
[dnl
  acx_lib_show_value="$[]$1[]_[]$4"
  acx_lib_show_value_ext="$[]$1[]_PUBLIC_[]$4"
  if test x$acx_lib_show_value_ext != x; then
    if test x$acx_lib_show_value != x; then
      acx_lib_show_value="$acx_lib_show_value "
    fi
    acx_lib_show_value="$acx_lib_show_value(+PUBLIC: $acx_lib_show_value_ext)"
  fi
  acx_lib_show_value_ext="$[]$1[]_BUILD_[]$4"
  if test x$acx_lib_show_value_ext != x; then
    if test x$acx_lib_show_value != x; then
      acx_lib_show_value="$acx_lib_show_value "
    fi
    acx_lib_show_value="$acx_lib_show_value(+BUILD: $acx_lib_show_value_ext)"
  fi
  if test x$5"$acx_lib_show_value" != x ; then
    acx_lib_show_txt="$acx_lib_show_txt
AS_HELP_STRING($2[]$3[:],[$acx_lib_show_value])"
  fi
])dnl _ACX_LIB_SHOW_NOTICE_FLAGS
dnl
dnl @synopsis ACX_LIB_SHOW_LIB(LIB,[HEADER],[LINE-PREFIX])
AC_DEFUN([ACX_LIB_SHOW_LIB],dnl
[_ACX_LIB_SHOW_LIB(m4_normalize([$1]),m4_normalize([$2]),m4_normalize([$3]))])
AC_DEFUN([_ACX_LIB_SHOW_LIB],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_DEFINED([$1],[],[m4_fatal([Unknown $1 object])])dnl
 acx_lib_show_txt="m4_ifval([$2], [$2], [Status for $1 _ACX_LIB_NAME([$1]):])"
 AS_CASE( [x"$acx_lib_have_[]$1"],
 [x],
   [acx_lib_show_txt="$acx_lib_show_txt
m4_ifval([$3], [$3], [  ])Not searched or created"],
 [dnl
   _ACX_LIB_IS_NEWOBJ([$1],
     [ m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWLIBS, [dnl
     	  m4_bmatch(_ACX_LIB_SUFFIX,
       	  [INFO],
	    [_ACX_LIB_SHOW_NOTICE([$1],[$3],[Link information],_ACX_LIB_SUFFIX)],
	  [LIBS_STATIC],
	    [_ACX_LIB_SHOW_NOTICE([$1],[$3],_ACX_LIB_SUFFIX,_ACX_LIB_SUFFIX)
	     _ACX_LIB_SHOW_NOTICE([$1],[$3],_ACX_LIB_SUFFIX[_ONLY],_ACX_LIB_SUFFIX[_ONLY])],
       	  [_ACX_LIB_SHOW_NOTICE([$1],[$3],_ACX_LIB_SUFFIX,_ACX_LIB_SUFFIX)]
     	  )
       ])
     ],[dnl
   _ACX_LIB_IS_EXTLIB([$1],[dnl
     AS_CASE( [x"$acx_lib_have_[]$1"],
     [xno],
       [acx_lib_show_txt="$acx_lib_show_txt
m4_ifval([$3], [$3], [  ])Not used or found"
       _ACX_LIB_VAR_PUSH_EXTLIB([$1])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Cpp flags],[CPPFLAGS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[C flags],[CFLAGS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[C++ flags],[CXXFLAGS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Linker flags],[LDFLAGS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Static linker flags],[LDFLAGS_STATIC])
       _ACX_LIB_VAR_POP_EXTLIB([$1])
       ],
     [xyes],
       [
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Version],[VERSION],[],[unknown])
       _ACX_LIB_SHOW_NOTICE_FLAGS([$1],[$3],[Cpp flags],[CPPFLAGS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Libs],[LIBS])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[Static libs],[LIBS_STATIC])
       _ACX_LIB_SHOW_NOTICE_FLAGS([$1],[$3],[C flags],[CFLAGS])
       _ACX_LIB_SHOW_NOTICE_FLAGS([$1],[$3],[C++ flags],[CXXFLAGS])
       _ACX_LIB_SHOW_NOTICE_FLAGS([$1],[$3],[Linker flags],[LDFLAGS])
       _ACX_LIB_SHOW_NOTICE_FLAGS([$1],[$3],[Static linker flags],[LDFLAGS_STATIC])
       _ACX_LIB_SHOW_NOTICE([$1],[$3],[PkgConfig],[PKGCONFIG])
       ],
     [x],
       [AC_MSG_ERROR(m4_location[: ACX_LIB@&t@_SHOW_LIB called for the external library $1 which is not defined. You must call ACX_LIB@&t@_CHECK_* or ACX_LIB@&t@_NOLIB for this library before])],
     [AC_MSG_ERROR(m4_location[: Invalid call to ACX_LIB@&t@_SHOW_LIB for $1])]
     )
   ])])
 ])
 AC_MSG_NOTICE([$acx_lib_show_txt])
])dnl ACX_LIB_SHOW_LIB
dnl
dnl @synopsis ACX_LIB_SHOW_EXTERNAL_LIBS()
AC_DEFUN([ACX_LIB_SHOW_EXTERNAL_LIBS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  _acx_lib_list=""
  for _acx_lib_lib in _ACX_LIB_LIST_EXTLIBS; do _acx_lib_list="$_acx_lib_list
  * $_acx_lib_lib" ; done
  AC_MSG_NOTICE([Libraries that can be used: $_acx_lib_list])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_LIST_EXTLIBS, [dnl
    ACX_LIB_SHOW_LIB(_ACX_LIB_SUFFIX)
  ])dnl
])
dnl
dnl @synopsis ACX_LIB_SHOW_NEW_LIBS()
AC_DEFUN([ACX_LIB_SHOW_NEW_LIBS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  _acx_lib_list=""
  for _acx_lib_lib in _ACX_LIB_LIST_NEWLIBS; do _acx_lib_list="$_acx_lib_list
  * $_acx_lib_lib" ; done
  AC_MSG_NOTICE([Libraries that can be created: $_acx_lib_list])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_LIST_NEWLIBS, [dnl
    ACX_LIB_SHOW_LIB(_ACX_LIB_SUFFIX)
  ])dnl
])
dnl
dnl @synopsis ACX_LIB_SHOW_NEW_PRGS()
AC_DEFUN([ACX_LIB_SHOW_NEW_PRGS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  _acx_lib_list=""
  for _acx_lib_lib in _ACX_LIB_LIST_NEWPRGS; do _acx_lib_list="$_acx_lib_list
  * $_acx_lib_lib" ; done
  AC_MSG_NOTICE([Programs that can be created: $_acx_lib_list])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_LIST_NEWPRGS, [dnl
    ACX_LIB_SHOW_LIB(_ACX_LIB_SUFFIX)
  ])dnl
])
dnl
dnl @synopsis ACX_LIB_SHOW()
AC_DEFUN([ACX_LIB_SHOW],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  ACX_LIB_SHOW_EXTERNAL_LIBS
  ACX_LIB_SHOW_NEW_LIBS
  ACX_LIB_SHOW_NEW_PRGS
])
dnl
dnl ====================================================================
dnl ====================================================================
dnl Macros to collect information about external libraries
dnl ------------------------------------------------------
dnl
dnl @synopsis _ACX_LIB_SET_PREFIX(PREFIX)
AC_DEFUN([_ACX_LIB_SET_PREFIX],
[ acx_lib_prefix=
  m4_ifval([$1], [dnl
    if [ test x"$1" != xyes ] && [ test x"$1" != xdefault ] && [ test x"$1" != x ] ; then
        acx_lib_prefix="$1"
    fi
  ])dnl
])
dnl
dnl @synopsis _ACX_LIB_MSG_CHECKING(LIB,[MORE INFO], [prefix], [PART])
AC_DEFUN([_ACX_LIB_MSG_CHECKING],
[dnl
   acx_lib_msg_prefix=
   if test x"$acx_lib_prefix" != x ; then
     acx_lib_msg_prefix="in $acx_lib_prefix"
   fi
   m4_case([$4],
   [running tests],
     [AC_MSG_NOTICE([looking for $1[]m4_ifval([$2],[ $2])[]m4_if([$3],[prefix],[ $acx_lib_msg_prefix])])],
   [AC_MSG_CHECKING([for $1[]m4_ifval([$2],[ $2])[]m4_if([$3],[prefix],[ $acx_lib_msg_prefix])[]m4_ifval([$4],[ ($4)])])])
])
dnl
dnl @synopsis _ACX_LIB_MSG_RESULT(LIB,[ACTION-IF-FOUND],[ACTION-IF-NOT-FOUND])
AC_DEFUN([_ACX_LIB_MSG_RESULT],
[dnl
  acx_lib_res="$[]$1[]_VERSION"
  case "$acx_lib_res" in
  unknown|"")
    acx_lib_res=yes
    ;;
  *) acx_lib_res="yes ($acx_lib_res)"
    ;;
  esac
  AS_IF([test $acx_lib_have_[]$1 = no], [
    acx_lib_res=no
  ])
  AC_MSG_RESULT([$acx_lib_res])
  AS_IF([test $acx_lib_have_[]$1 = no], [:
    m4_ifval([$3], [$3])dnl
  ],[:
    m4_ifval([$2], [$2])dnl
  ])
])
dnl
dnl @synopsis _ACX_LIB_CHECK_VAR(LIB)
AC_DEFUN([_ACX_LIB_CHECK_VAR],
[dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_EXTLIB([$1],[],[dnl
_ACX_LIB_IS_DEFINED([$1],
[m4_fatal([$1 already defined. $1 cannot be redefined as a external library])],
[m4_define([_ACX_LIB_LIST_EXTLIBS],_ACX_LIB_LIST_EXTLIBS[ $1])dnl
dnl AC_ARG_VAR([$1][_CFLAGS], [C compiler flags for $1, overriding auto-detection])dnl
dnl AC_ARG_VAR([$1][_LIBS], [linker flags for $1, overriding auto-detection])dnl
dnl AC_ARG_VAR([$1][_LIBS_STATIC], [static linker flags for $1, overriding auto-detection, default to $1_LIBS])dnl
])])
  AS_CASE([x"$acx_lib_state_[]$1"],
    [xlinked], [AC_MSG_ERROR(m4_location[: cannot change $1 as this library has already been linked to another object])],
    [x], [
      m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [AC_SUBST([$1]_[]_ACX_LIB_SUFFIX)])
      acx_lib_state_[]$1=defined],
    [xdefined], [
      AS_CASE([x"$acx_lib_have_[]$1"],
      [xno],[],
      [xyes], [AC_MSG_WARN([redefining $1 library that has been already found])],
      [AC_MSG_WARN([redefining $1 library])])
    ],
    [AC_MSG_ERROR(m4_location[: Strange kind of state for $1 ($acx_lib_state_[]$1). Aborting.])]
   )
])
dnl
dnl @synopsis ACX_LIB_NOLIB(LIB)
dnl 
AC_DEFUN([ACX_LIB_NOLIB],
[AC_REQUIRE([_ACX_LIB_INIT])
  _ACX_LIB_CHECK_VAR([$1])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
    $1[]_[]_ACX_LIB_SUFFIX=""
  ])
  acx_lib_have_[]$1=no
  AC_MSG_NOTICE([not using $1 library])
])dnl ACX_LIB_NOLIB
dnl
dnl **************
dnl
dnl @synopsis _ACX_LIB_FOUND([Id=value, ...])
AC_DEFUN([_ACX_LIB_FOUND],
[m4_ifdef([_ACX_LIB_ASSIGN_LIST], [m4_undefine([_ACX_LIB_ASSIGN_LIST])])dnl
  acx_lib_found=yes
  _ACX_LIB_FOUND_ASSIGN($@)dnl
])
AC_DEFUN([_ACX_LIB_FOUND_ASSIGN],
[m4_case($#,
 [0],[],
 [1],[_ACX_LIB_FOUND_ASSIGN_ARG([$1],[ACX_LIB_FOUND])],
 [_ACX_LIB_FOUND_ASSIGN_ARG([$1],[ACX_LIB_FOUND])dnl
  _ACX_LIB_FOUND_ASSIGN(m4_shift($@))])dnl
])
dnl
AC_DEFUN([_ACX_LIB_FOUND_ASSIGN_ARG],
[m4_bmatch(m4_toupper($1),
  [^[ ]*\(VERSION\|C\(PP\|XX\)?FLAGS\|\(LDFLAGS\|LIBS\)\([ _-]STATIC\)?\|PKGCONFIG\)[ ]*=],dnl
   [_ACX_LIB_ASSIGN_ARG_EXTRACT([$1],[],[$2])],
  [^.*=], [m4_fatal([Unknown id ']m4_bpatsubst($1,[^[ ]*\(.*[^ ]\)[ ]*=.*],[\1])[' in $2 arguments])],
  [^[ ]*$], [],
  [m4_fatal([Wrong syntax '$1' in $2 arguments])]
 )
])
AC_DEFUN([_ACX_LIB_ASSIGN_ARG_EXTRACT],
[_ACX_LIB_ASSIGN_ARG_REAL([m4_translit(m4_bpatsubst(m4_toupper($1),dnl
[^[ ]*\(.*[^ ]\)[ ]*=.*],[\1]),[ -],[__])],dnl
[m4_bpatsubst([$1],[^[^=]*=[ ]*],[])],[$2],[$3])])
AC_DEFUN([_ACX_LIB_ASSIGN_ARG_REAL],
[m4_ifdef([_ACX_LIB_ASSIGN_LIST],dnl
   [m4_pushdef([_ACX_LIB_ASSIGN_LIST_OLD],m4_defn([_ACX_LIB_ASSIGN_LIST]))],
   [m4_define([_ACX_LIB_ASSIGN_LIST_OLD],[])])dnl
m4_append_uniq([_ACX_LIB_ASSIGN_LIST],$1,[ ])dnl
m4_if(_ACX_LIB_ASSIGN_LIST_OLD,_ACX_LIB_ASSIGN_LIST,[
  m4_fatal([Id ']$1[' used several times in $4 arguments])
 ])dnl
dnl m4_errprintn([list ($4): ]_ACX_LIB_ASSIGN_LIST)dnl
acx_lib_[]$1[]$3=$2
])
dnl
dnl @synopsis _ACX_LIB_NOTFOUND()
AC_DEFUN([_ACX_LIB_NOTFOUND],
[ acx_lib_found=no
])
dnl
dnl @synopsis _ACX_LIB_TRYFOUND(CODE-THAT-SHOULD-CALL-ACX_LIB_[NOT]FOUND)
AC_DEFUN([_ACX_LIB_TRYFOUND],
[dnl
    acx_lib_found=trying
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      unset acx_lib_[]_ACX_LIB_SUFFIX
    ])
    m4_pushdef([ACX_LIB_FOUND],m4_defn([_ACX_LIB_FOUND]))
    m4_pushdef([ACX_LIB_NOTFOUND],m4_defn([_ACX_LIB_NOTFOUND]))
    $1
    m4_popdef([ACX_LIB_FOUND])
    m4_popdef([ACX_LIB_NOTFOUND])
    case "$acx_lib_found" in
    no|yes) ;;
    trying)
      AC_MSG_WARN(m4_location[: assuming [ACX_LIB@&t@_NOTFOUND] (neither [ACX_LIB@&t@_FOUND] nor [ACX_LIB@&t@_NOTFOUND] called)])
      acx_lib_found=no
      ;;
    *) AC_MSG_ERROR(m4_location[: Invalid value for acx_lib_found: $acx_lib_found])
      ;;
    esac
])
dnl
dnl @synopsis _ACX_LIB_CHECK_RESULTS(LIB)
AC_DEFUN([_ACX_LIB_CHECK_RESULTS],
[dnl
  AS_CASE(["$acx_lib_found"],
  [no], [],
  [yes], [dnl
    if test x"${acx_lib_LIBS_STATIC+set}" = x; then
      acx_lib_LIBS_STATIC="$acx_lib_LIBS"
    fi
    if test x"${acx_lib_LDFLAGS_STATIC+set}" = x; then
      acx_lib_LDFLAGS_STATIC="$acx_lib_LDFLAGS"
    fi
    if test x"$acx_lib_VERSION" = x ; then
      acx_lib_VERSION=unknown
    fi
    if test x"${acx_lib_CFLAGS+set}" = xset && test x"${acx_lib_CPPFLAGS+set}" = x; then
      AC_MSG_WARN([for $1, CFlags are set but not CppFlags. This is usually an error.])
      AC_MSG_WARN([Set empty CppFlags in configure.ac to hide this warning.])
    elif test x"${acx_lib_CXXFLAGS+set}" = xset && test x"${acx_lib_CPPFLAGS+set}" = x; then
      AC_MSG_WARN([for $1, CxxFlags are set but not CppFlags. This is usually an error.])
      AC_MSG_WARN([Set empty CppFlags in configure.ac to hide this warning.])
    fi
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      $1[]_[]_ACX_LIB_SUFFIX="$acx_lib_[]_ACX_LIB_SUFFIX"
    ])
  ],
  [AC_MSG_ERROR(m4_location[: No result for $1. Unable to compute variables value.
  ACX_LIB@&t@_FOUND or ACX_LIB@&t@_NOTFOUND must be called. See documentation for ACX_LIB@&t@_CHECK_*])])
  acx_lib_have_[]$1=$acx_lib_found
  unset acx_lib_found
])
dnl
dnl **************
dnl
dnl @synopsis ACX_LIB_CHECK_PKGCONFIG(VARIABLE-PREFIX, [prefix], MODULES, [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
AC_DEFUN([ACX_LIB_CHECK_PKGCONFIG],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
  _ACX_LIB_CHECK_VAR([$1])
  _ACX_LIB_SET_PREFIX([$2])
  m4_ifval([$2], [dnl
    _ACX_LIB_VAR_PUSH([PKG_CONFIG_PATH])
    if [ test x"$acx_lib_prefix" != x ] ; then
        PKG_CONFIG_PATH="$acx_lib_prefix/lib/pkgconfig$PATH_SEPARATOR$acx_lib_prefix/share/pkgconfig$PATH_SEPARATOR$PKG_CONFIG_PATH"
        export PKG_CONFIG_PATH
    fi
  ])dnl
  _ACX_LIB_MSG_CHECKING([$1], [($3) with pkg-config], [prefix], [running tests])
  _ACX_LIB_TRYFOUND([
    PKG_CHECK_EXISTS([$3], [dnl
      dnl Call PKG_CHECK_EXISTS to really check for the module
      dnl else, PKG_CHECK_MODULES can succeed with only $1_LIBS
      dnl and $1_CFLAGS defined and no .pc file present
      PKG_CHECK_MODULES([$1], [$3], [dnl
        acx_lib_CPPFLAGS="$[]$1[]_CFLAGS"
        acx_lib_LIBS="$[]$1[]_LIBS"
        acx_lib_PKGCONFIG="$3"
        _PKG_CONFIG([$1][_LIBS_STATIC], [libs --static], [$3])
        AS_IF([test x"$pkg_failed" = "yes"],
    	       [AC_MSG_WARN([Using $1[]_LIBS for $1[]_LIBS_STATIC])
                acx_lib_LIBS_STATIC="$acx_lib_LIBS"],
            [acx_lib_LIBS_STATIC=$pkg_cv_[]$1[]_LIBS_STATIC])
        _PKG_CONFIG([$1][_VERSION], [modversion], [$3])
        if test x"$pkg_failed" != "yes" ; then
          acx_lib_VERSION="$pkg_cv_[]$1[]_VERSION"
        fi
        _ACX_LIB_FOUND()
      ], [dnl
        ACX_LIB_NOTFOUND
      ])
    ], [ACX_LIB_NOTFOUND])
  ])
  m4_ifval([$2], [dnl
    _ACX_LIB_VAR_POP([PKG_CONFIG_PATH])
  ])dnl
  _ACX_LIB_CHECK_RESULTS([$1])
  _ACX_LIB_MSG_CHECKING([$1], [($3) with pkg-config], [], [results])
  _ACX_LIB_MSG_RESULT([$1], [$4], [$5])
])
dnl
dnl @synopsis ACX_LIB_CHECK_PLAIN(VARIABLE-PREFIX, [prefix], TESTS, [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
AC_DEFUN([ACX_LIB_CHECK_PLAIN],
[AC_REQUIRE([_ACX_LIB_INIT])
  _ACX_LIB_CHECK_VAR([$1])
  _ACX_LIB_SET_PREFIX([$2])

  _ACX_LIB_MSG_CHECKING([$1],[], [prefix], [running tests])

  _ACX_LIB_VAR_PUSH([CFLAGS])
  _ACX_LIB_VAR_PUSH([CPPFLAGS])
  _ACX_LIB_VAR_PUSH([CXXFLAGS])
  _ACX_LIB_VAR_PUSH([LIBS])

  if test x"$acx_lib_prefix" = x ; then
    ACX_LIB_CPPFLAGS=
    ACX_LIB_LIBS=
  else
    ACX_LIB_CPPFLAGS="-I$acx_lib_prefix/include"
    ACX_LIB_LIBS="-L$acx_lib_prefix/lib"
  fi

  # Running user tests
  _ACX_LIB_TRYFOUND([$3])

  _ACX_LIB_VAR_POP([CFLAGS])
  _ACX_LIB_VAR_POP([CPPFLAGS])
  _ACX_LIB_VAR_POP([CXXFLAGS])
  _ACX_LIB_VAR_POP([LIBS])

  _ACX_LIB_CHECK_RESULTS([$1])

  _ACX_LIB_MSG_CHECKING([$1], [], [], [results])
  _ACX_LIB_MSG_RESULT([$1], [$4], [$5])
])
dnl
dnl @synopsis _ACX_LIB_PROG_LIBRARY_CONFIG(VARIABLE-PREFIX, [prefix], library-config)
AC_DEFUN([_ACX_LIB_PROG_LIBRARY_CONFIG],
[dnl
AC_ARG_VAR([$1][_CONFIG], [path to $3 utility])dnl
if test "x$ac_cv_env_[]$1[]_CONFIG_set" != "xset"; then
   	_acx_lib_path="${acx_lib_prefix:+$acx_lib_prefix/bin$PATH_SEPARATOR}$PATH"
        AC_PATH_TOOL([$1][_CONFIG], [$3], [], [$_acx_lib_path])
fi
])
dnl
dnl @synopsis _ACX_LIB_LIBRARY_CONFIG(VARIABLE-PREFIX, VARIABLE_SUFFIX, OPTIONS)
AC_DEFUN([_ACX_LIB_LIBRARY_CONFIG],
[dnl
    AC_MSG_CHECKING([$1[]_[]$2])
    if test x"${acx_lib_[]$2[]+set}" = xset; then
        AC_MSG_RESULT([$acx_lib_[]$2 (supplied value)])
    else
        acx_lib_[]$2=`$[]$1[]_CONFIG $acx_lib_[]$2[]_CONFIG_OPTION $acx_lib_MODULES_CONFIG_OPTION 2>/dev/null`
	acx_lib_status=$?
	if test $acx_lib_status = 0 ; then
	    AC_MSG_RESULT([$acx_lib_[]$2])
	else
	    AC_MSG_RESULT([no: error while running "$[]$1[]_CONFIG $acx_lib_[]$2[]_CONFIG_OPTION $acx_lib_MODULES_CONFIG_OPTION"])
	    acx_lib_failed=yes
        fi
    fi
])
dnl
dnl
AC_DEFUN([_ACX_LIB_CHECK_CONFIG_ASSIGN],
[m4_case($#,
 [0],[],
 [1],[_ACX_LIB_CHECK_CONFIG_ASSIGN_ARG([$1],[ACX_LIB_CHECK_CONFIG])],
 [_ACX_LIB_CHECK_CONFIG_ASSIGN_ARG([$1],[ACX_LIB_CHECK_CONFIG])dnl
  m4_if(m4_toupper(m4_normalize($1)),[STOP],,[_ACX_LIB_CHECK_CONFIG_ASSIGN(m4_shift($@))])])dnl
])
dnl
AC_DEFUN([_ACX_LIB_CHECK_CONFIG_ASSIGN_ARG],
[m4_define([_ACX_LIB_CHECK_CONFIG_COUNTER], m4_incr(_ACX_LIB_CHECK_CONFIG_COUNTER))dnl
m4_bmatch(m4_toupper($1),
  [^[ ]*\(MODULES\|VERSION\|C\(PP\|XX\)?FLAGS\|\(LDFLAGS\|LIBS\)\([ _-]STATIC\)?\)[ ]*=],dnl
   [_ACX_LIB_ASSIGN_ARG_EXTRACT([$1],[[_CONFIG_OPTION]],[$2])],
  [^[A-Z]*=], [m4_fatal([Unknown id ']m4_bpatsubst($1,[^[ ]*\(.*[^ ]\)[ ]*=.*],[\1])[' in $2 arguments])],
  [^[ ]*\(STOP\)?[ ]*$], [],
  [m4_fatal([Wrong syntax '$1' in $2 arguments. Do you forget the STOP argument?])]
 )dnl
])
dnl
dnl @synopsis ACX_LIB_CHECK_CONFIG(VARIABLE-PREFIX, [prefix], library-config,
dnl [id=value, ...], stop, [TESTS = ACX_LIB_FOUND()],
dnl [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
AC_DEFUN([ACX_LIB_CHECK_CONFIG],
[AC_REQUIRE([_ACX_LIB_INIT])
  _ACX_LIB_CHECK_VAR([$1])
  _ACX_LIB_SET_PREFIX([$2])
  m4_define([_ACX_LIB_CHECK_CONFIG_COUNTER], [4])dnl
  m4_ifdef([_ACX_LIB_ASSIGN_LIST], [m4_undefine([_ACX_LIB_ASSIGN_LIST])])dnl
  _ACX_LIB_CHECK_CONFIG_ASSIGN(m4_shiftn(3,$@))
  m4_ifdef([_ACX_LIB_ASSIGN_LIST],[],[
    m4_fatal([ACX_LIB_CHECK_CONFIG called without any ID="lib-config-options" parameters])])dnl
  m4_define([_ACX_LIB_CHECK_CONFIG_LIST],m4_defn([_ACX_LIB_ASSIGN_LIST]))dnl

  _ACX_LIB_MSG_CHECKING([$1], [with $3], [prefix], [running tests])
  _ACX_LIB_PROG_LIBRARY_CONFIG([$1], [$acx_lib_prefix], [$3])
  if test x"$[]$1[]_CONFIG" = x ; then
    _ACX_LIB_TRYFOUND([ACX_LIB_NOTFOUND])
  else
    acx_lib_failed=no
    _ACX_LIB_VAR_PUSH([$1[]_VERSION])
    _ACX_LIB_TRYFOUND([
      m4_if(m4_index([ ]_ACX_LIB_CHECK_CONFIG_LIST[ ],[ VERSION ]),-1,[
        acx_lib_VERSION="$[]$1[]_VERSION"
      ], [
        _ACX_LIB_LIBRARY_CONFIG([$1], [VERSION])
        $1[]_VERSION="$acx_lib_VERSION"
      ])
      m4_ifval(m4_normalize(_ACX_LIB_argn(_ACX_LIB_CHECK_CONFIG_COUNTER, $@)),
         _ACX_LIB_argn(_ACX_LIB_CHECK_CONFIG_COUNTER, $@), [ACX_LIB_FOUND()])])
    m4_define([_ACX_LIB_CHECK_CONFIG_COUNTER], m4_incr(_ACX_LIB_CHECK_CONFIG_COUNTER))dnl
    if test $acx_lib_found = yes ; then
      _ACX_LIB_VAR_POP([$1[]_VERSION])
      m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_CHECK_CONFIG_LIST, [dnl
	m4_case(_ACX_LIB_SUFFIX,
	[VERSION],[],
	[MODULES],[],
	[_ACX_LIB_LIBRARY_CONFIG([$1], _ACX_LIB_SUFFIX)])
      ])
      if test $acx_lib_failed = yes ; then
        acx_lib_found=no
      fi
    else
      _ACX_LIB_VAR_POP([$1[]_VERSION])
    fi
  fi
  _ACX_LIB_CHECK_RESULTS([$1])
  _ACX_LIB_MSG_CHECKING([$1], [with $3], [], [results])
  _ACX_LIB_MSG_RESULT([$1], _ACX_LIB_argn(_ACX_LIB_CHECK_CONFIG_COUNTER, $@),
                            _ACX_LIB_argn(m4_incr(_ACX_LIB_CHECK_CONFIG_COUNTER), $@))
])
dnl ====================================================================
dnl ====================================================================
dnl Macros to manage new libraries
dnl ------------------------------
dnl
dnl @synopsis ACX_LIB_LINKWITH_IFFOUND(NEW, SEARCHED, WORK)
AC_DEFUN([_ACX_LIB_LINKWITH_IFFOUND],
[dnl
  _ACX_LIB_IS_LIB([$2],
    [_ACX_LIB_IS_NEWLIB([$2],[_ACX_LIB_IS_NEWEXTLIB([$2],[],
        [m4_fatal([Cannot link $1 with $2: $2 is a new library without link parameters])])])],
    [m4_fatal([Cannot link $1 with $2: $2 is not a library])])dnl
  AS_CASE([x"$acx_lib_state_[]$2"],
   [xdefined], [acx_lib_state_[]$2=linked],
   [xlinked], [],
   [x], [AC_MSG_ERROR(m4_location[: ACX_LIB@&t@_CHECK_* or ACX_LIB@&t@_NOLIB must be called before linking $2])],
   [AC_MSG_ERROR(m4_location[: Strange kind of state for $2 ($acx_lib_state_[]$2). Aborting.])]
  )
  _ACX_LIB_IS_EXTLIB([$2],[dnl
    AS_CASE([x"$acx_lib_have_[]$2"],
      [xno], [_ACX_LIB_VAR_PUSH_EXTLIB([$2])],
      [xyes],
        [],
      [x],
        [ AC_MSG_ERROR(m4_location[: when linking $1 with $2, $2 must already be defined with ACX_LIB@&t@_CHECK_* or ACX_LIB@&t@_NOLIB]) ],
      [ AC_MSG_ERROR(m4_location[: Invalid library $2 to link in $1 ($acx_lib_have_$2)]) ]
    )
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      m4_bmatch(_ACX_LIB_SUFFIX,
        [VERSION],[],
	[PKGCONFIG], [],
	[FLAGS\|LIBS], [
           _ACX_LIB_VAR_PUSH([$2[]_CONFIG_[]_ACX_LIB_SUFFIX],
	       [$[]$2[]_[]_ACX_LIB_SUFFIX])
	   _ACX_LIB_VAR_PUSH([$2[]_CONFIG_[]_ACX_LIB_SUFFIX[]_NOPUBLIC],
	       [$[]$2[]_[]_ACX_LIB_SUFFIX])],
	[m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_LINKWITH_IFFOUND])]
      )
    ])dnl
    $3
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      m4_bmatch(_ACX_LIB_SUFFIX,
        [VERSION],[],
	[PKGCONFIG], [],
	[FLAGS\|LIBS], [
           _ACX_LIB_VAR_POP([$2[]_CONFIG_[]_ACX_LIB_SUFFIX])
	   _ACX_LIB_VAR_POP([$2[]_CONFIG_[]_ACX_LIB_SUFFIX[]_NOPUBLIC])],
	[m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_LINKWITH_IFFOUND])]
      )
    ])dnl
    AS_CASE([x"$acx_lib_have_[]$2"],
      [xno], [_ACX_LIB_VAR_POP_EXTLIB([$2])]
    )
  ],[dnl
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      _ACX_LIB_VAR_PUSH([$2[]_[]_ACX_LIB_SUFFIX])
      m4_bmatch(_ACX_LIB_SUFFIX,
        [VERSION],[$2[]_VERSION="internal"],
	[PKGCONFIG], [],
	[FLAGS], [$2[]_[]_ACX_LIB_SUFFIX[]_PUBLIC=""],
	[LIBS], [],
	[m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_LINKWITH_IFFOUND])]
      )
    ])dnl
    $3
    m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_EXTLIBS, [dnl
      _ACX_LIB_VAR_POP([$2[]_[]_ACX_LIB_SUFFIX])
    ])dnl
  ])
])
dnl
dnl @synopsis ACX_LIB_LINKWITH_BASE(NEW, SEARCHED, [info])
AC_DEFUN([_ACX_LIB_LINKWITH_BASE],
[dnl
  AC_MSG_NOTICE([Recording that $1 is linked with $2]m4_ifval([$3],[ ($3 linkage)]))
 _ACX_LIB_IS_EXTLIB([$2],[dnl
  AS_CASE([x"$acx_lib_have_$2"],
   [xyes],[dnl
     _acx_lib_linkwith_version="$[]$2[]_VERSION"
     if test x"$_acx_lib_linkwith_version" = xunknown ; then
       _acx_lib_linkwith_version="$2"
     else
       _acx_lib_linkwith_version="$2 ($_acx_lib_linkwith_version)"
     fi
   ],
   [_acx_lib_linkwith_version="NOT $2"])],
 [_acx_lib_linkwith_version="$2 (local)"])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWPRGS, [dnl
    m4_bmatch(_ACX_LIB_SUFFIX,
      [INFO], [dnl
        _ACX_LIB_VAR_ADD([$1], [INFO],
	  [$_acx_lib_linkwith_version[]_ACX_LIB_IS_NEWLIB(
	       [$1],[m4_if($3,[Private],[],[ @<:@$3@:>@])])], [, ])],
      [BUILD_.*FLAGS], [dnl
 	_ACX_LIB_IS_EXTLIB([$1],[dnl
  	  _ACX_LIB_VAR_ADD([$1], _ACX_LIB_SUFFIX,
	    [$[]$2[]_[]m4_bpatsubst(_ACX_LIB_SUFFIX, [BUILD_], [BUILDONLY_])])])
        _ACX_LIB_VAR_ADD([$1], _ACX_LIB_SUFFIX,
	  [$[]$2[]_[]m4_bpatsubst(_ACX_LIB_SUFFIX, [BUILD_], [CONFIG_])[]_NOPUBLIC])],
      [BUILD_LIBS], [dnl
        _ACX_LIB_VAR_ADD([$1], _ACX_LIB_SUFFIX,
	  [$[]$2[]_[]m4_bpatsubst(_ACX_LIB_SUFFIX, [BUILD_], [CONFIG_])[]_NOPUBLIC])],
      [m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_LINKWITH_BASE])]
    )
  ])
 _ACX_LIB_IS_NEWLIB([$1],[dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWLIBS_ONLY, [dnl
    m4_bmatch(_ACX_LIB_SUFFIX,
      [PKGCONFIG_CFLAGS], [dnl
        _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CXXFLAGS_PUBLIC])
  	_ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CPPFLAGS_PUBLIC])
  	_ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CFLAGS_PUBLIC])],
      [PKGCONFIG_LIBS], [dnl
        _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS], [$[]$2[]_LDFLAGS_PUBLIC])],
dnl PKGCONFIG_LIBS_PRIVATE should not include flag already present in PKGCONFIG_LIBS...
      [PKGCONFIG_LIBS_PRIVATE], [dnl
        _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$[]$2[]_LDFLAGS_STATIC_PUBLIC])],
      [PKGCONFIG_REQUIRES], [],
      [CONFIG_.*FLAGS], [dnl
        _ACX_LIB_VAR_ADD([$1], _ACX_LIB_SUFFIX,
	  [$[]$2[]_[]m4_bpatsubst(_ACX_LIB_SUFFIX, [CONFIG_], [])[]_PUBLIC])],
      [CONFIG_LIBS], [],
      [m4_fatal([Unmanaged suffix ]_ACX_LIB_SUFFIX[ in _ACX_LIB_LINKWITH_BASE])]
    )
  ])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS_STATIC], [$[]$2[]_CONFIG_LDFLAGS_STATIC])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_LIBS_STATIC], [$[]$2[]_CONFIG_LIBS_STATIC])
  _ACX_LIB_IS_NEWEXTLIB([$1],[dnl
    _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS_STATIC_NOPUBLIC], [$[]$2[]_CONFIG_LDFLAGS_STATIC_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_LIBS_STATIC_NOPUBLIC], [$[]$2[]_CONFIG_LIBS_STATIC_NOPUBLIC])
  ])dnl
 ])dnl
])

AC_DEFUN([_ACX_LIB_LINKWITH_PRIVATE],
[dnl
  _ACX_LIB_LINKWITH_BASE([$1],[$2],[m4_ifval([$3],[$3],[Private])])
  if test x"$[]$2[]_PKGCONFIG" != x ; then
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_REQUIRES_PRIVATE], [$[]$2[]_PKGCONFIG], [, ])
  else
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$[]$2[]_LDFLAGS_STATIC])
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$[]$2[]_LIBS_STATIC])
  fi
])

AC_DEFUN([_ACX_LIB_LINKWITH_PUBLIC],
[dnl
  _ACX_LIB_LINKWITH_BASE([$1], [$2], [Public])
  if test x"$[]$2[]_PKGCONFIG" != x ; then
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_REQUIRES], [$[]$2[]_PKGCONFIG], [, ])
  else
dnl PKGCONFIG_LIBS_PRIVATE should not include flag already present in PKGCONFIG_LIBS...
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS], [$[]$2[]_LDFLAGS])
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$[]$2[]_LDFLAGS_STATIC])
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS], [$[]$1[]_LIBS])
    for _acx_lib_var_lib in $[]$2[]_LIBS_STATIC ; do
      case "$_acx_lib_var_lib" in
      -l*|-L*)
	case " $[]$2[]_LIBS " in
	*\ "$_acx_lib_var_lib"\ *)
	  ;; # already in public libs
	*)
 	  _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$_acx_lib_var_lib])
	  ;;
	esac
	;;
      *)
        _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$_acx_lib_var_lib])
	;;
      esac
    done
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CPPFLAGS])   
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CFLAGS])   
    _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CXXFLAGS])   
  fi
  _ACX_LIB_VAR_ADD([$1], [CONFIG_LIBS], [$[]$2[]_CONFIG_LIBS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS], [$[]$2[]_CONFIG_LDFLAGS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CPPFLAGS], [$[]$2[]_CONFIG_CPPFLAGS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CFLAGS], [$[]$2[]_CONFIG_CFLAGS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CXXFLAGS], [$[]$2[]_CONFIG_CXXFLAGS])
  _ACX_LIB_IS_NEWEXTLIB([$1],[dnl
    _ACX_LIB_VAR_ADD([$1], [CONFIG_LIBS_NOPUBLIC], [$[]$2[]_CONFIG_LIBS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_LDFLAGS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CPPFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CPPFLAGS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CFLAGS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CXXFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CXXFLAGS_NOPUBLIC])
  ])
])

AC_DEFUN([_ACX_LIB_LINKWITH_PUBLIC_HEADERS],
[dnl
  _ACX_LIB_LINKWITH_PRIVATE([$1], [$2], [Private but headers])
  _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CPPFLAGS])   
  _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CFLAGS])   
  _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$[]$2[]_CONFIG_CXXFLAGS])   
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CPPFLAGS], [$[]$2[]_CONFIG_CPPFLAGS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CFLAGS], [$[]$2[]_CONFIG_CFLAGS])
  _ACX_LIB_VAR_ADD([$1], [CONFIG_CXXFLAGS], [$[]$2[]_CONFIG_CXXFLAGS])
  _ACX_LIB_IS_NEWEXTLIB([$1],[dnl
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CPPFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CPPFLAGS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CFLAGS_NOPUBLIC])
    _ACX_LIB_VAR_ADD([$1], [CONFIG_CXXFLAGS_NOPUBLIC], [$[]$2[]_CONFIG_CXXFLAGS_NOPUBLIC])
  ])
])

AC_DEFUN([ACX_LIB_LINK],dnl
[_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_NEWOBJ([$1],[],
[m4_fatal([$1 is not defined as a new library or program. Use ACX_LIB][_NEW_*])])
  AS_CASE([x"$acx_lib_state_[]$1"],
   [xdefined], [],
   [xlinked], [AC_MSG_ERROR(m4_location[: cannot change $1 after being already linked])],
   [AC_MSG_ERROR(m4_location[: Strange kind of state for $1 ($acx_lib_state_[]$1). Aborting.])]
  )
 m4_if($#,[2],
 [_ACX_LIB_LINK([$1],[private],[$2])],
 [_ACX_LIB_LINK($@)]
 )
])

AC_DEFUN([_ACX_LIB_LINK],dnl
[m4_case($#,dnl
  [1],[],
  [2],[m4_ifval([$2],[m4_fatal([Bad number of parameters ($#: $@) for ACX_LIB_LINK])])],
  [dnl
m4_bmatch(m4_toupper(m4_normalize([$2])),dnl
[^PRIVATE$], [dnl
  m4_foreach_w([_ACX_LIB_LIBNAME], [$3], [dnl
    _ACX_LIB_ENFORCE_NAME(_ACX_LIB_LIBNAME)dnl
    _ACX_LIB_LINKWITH_IFFOUND([$1], _ACX_LIB_LIBNAME, [dnl
      _ACX_LIB_LINKWITH_PRIVATE([$1], _ACX_LIB_LIBNAME)
    ])
  ])
],
[^PUBLIC$], [dnl
  _ACX_LIB_IS_NEWPRG([$1],
    [m4_fatal([$1 is a program: only private linkage is supported])])
  m4_foreach_w([_ACX_LIB_LIBNAME], [$3], [dnl
    _ACX_LIB_ENFORCE_NAME(_ACX_LIB_LIBNAME)dnl
    _ACX_LIB_LINKWITH_IFFOUND([$1], _ACX_LIB_LIBNAME, [dnl
      _ACX_LIB_LINKWITH_PUBLIC([$1], _ACX_LIB_LIBNAME)
    ])
  ])
],
[^HALF[- ]PRIVATE$], [dnl
  _ACX_LIB_IS_NEWPRG([$1],
    [m4_fatal([$1 is a program: only private linkage is supported])])
  m4_foreach_w([_ACX_LIB_LIBNAME], [$3], [dnl
    _ACX_LIB_ENFORCE_NAME(_ACX_LIB_LIBNAME)dnl
    _ACX_LIB_LINKWITH_IFFOUND([$1], _ACX_LIB_LIBNAME, [dnl
      _ACX_LIB_LINKWITH_PUBLIC_HEADERS([$1], _ACX_LIB_LIBNAME)
    ])
  ])
],[
  m4_fatal([Unknown MODE '$2' in ACX_LIB_LINK])
])
_ACX_LIB_LINK([$1], m4_shiftn(3,$@))
])])

AC_DEFUN([ACX_LIB_NEW_LIB],dnl
[_ACX_LIB_NEW_LIB(m4_normalize([$1]),m4_normalize([$2]),m4_normalize([$3]))])
AC_DEFUN([_ACX_LIB_NEW_LIB],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_DEFINED([$1],
[m4_fatal([$1 already defined. $1 cannot be redefined with ACX_LIB_NEW_LIB])],
[m4_define([_ACX_LIB_LIST_NEWLIBS],_ACX_LIB_LIST_NEWLIBS[ $1])])
  acx_lib_have_[]$1=newlib
  acx_lib_state_[]$1=defined
  m4_ifval([$2],[m4_define([_ACX_LIB_LIST_NEWEXTLIBS],_ACX_LIB_LIST_NEWEXTLIBS[ $1])dnl
    $1[]_PKGCONFIG="$3"
    $1[]_PKGCONFIG_LIBS="$2"
    acx_lib_have_[]$1=newext
  ])
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWLIBS, [dnl
    $1[]_[]_ACX_LIB_SUFFIX=""
    m4_bmatch(_ACX_LIB_SUFFIX,
      [INFO], [],
      [^CONFIG_.*FLAGS], [dnl
        AC_SUBST([$1]_[]_ACX_LIB_SUFFIX)
	_ACX_LIB_IS_NEWEXTLIB([$1],[dnl
          $1[]_[]_ACX_LIB_SUFFIX[]_NOPUBLIC=""
        ])dnl
      ],
      [CONFIG_LIBS\(_STATIC\)?], [dnl
        AC_SUBST([$1]_[]_ACX_LIB_SUFFIX)
	_ACX_LIB_IS_NEWEXTLIB([$1],[dnl
          $1[]_[]m4_bpatsubst(_ACX_LIB_SUFFIX, [CONFIG_], [LOCALBUILD_])=""	
          $1[]_[]_ACX_LIB_SUFFIX="$2"	
        ])dnl
      ],
      [AC_SUBST([$1]_[]_ACX_LIB_SUFFIX)]
    )
  ])
])

AC_DEFUN([ACX_LIB_NEW_PRG],dnl
[_ACX_LIB_NEW_PRG(m4_normalize([$1]),m4_normalize([$2]))])
AC_DEFUN([_ACX_LIB_NEW_PRG],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_DEFINED([$1],
[m4_fatal([$1 already defined. $1 cannot be redefined with ACX_LIB_NEW_PRG])],
[m4_define([_ACX_LIB_LIST_NEWPRGS],_ACX_LIB_LIST_NEWPRGS[ $1])])
  acx_lib_have_[]$1=newprg
  acx_lib_state_[]$1=defined
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWPRGS, [dnl
    $1[]_[]_ACX_LIB_SUFFIX=""
    m4_bmatch(_ACX_LIB_SUFFIX,
      [INFO], [],
      [AC_SUBST([$1]_[]_ACX_LIB_SUFFIX)]
    )
  ])
  ACX_LIB_LINK([$1],[private],[$2])
])

dnl ====================================================================
dnl ====================================================================
dnl Macros to add new flags
dnl ------------------------------
dnl
dnl
dnl @synopsis ACX_LIB_ADD_BUILD_FLAGS(VARIABLE-PREFIX, [CPPFLAGS], [CFLAGS], [CXXFLAGS], [LDFLAGS], [LDFLAGS_STATIC])
AC_DEFUN([ACX_LIB_ADD_BUILD_FLAGS],dnl
[_ACX_LIB_ADD_BUILD_FLAGS(m4_normalize([$1]),m4_normalize([$2]),
m4_normalize([$3]),m4_normalize([$4]),m4_normalize([$5]),dnl
m4_normalize([$6]))])
AC_DEFUN([_ACX_LIB_ADD_BUILD_FLAGS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_DEFINED([$1], [dnl
 AS_CASE([x"$acx_lib_state_[]$1"],
 [xdefined], [],
 [xmerged], [AC_MSG_ERROR(m4_location[: the $1 library has already be linked. Cannot add new flags])],
 [x], [AC_MSG_ERROR(m4_location[: $1 has not been defined yet. Cannot add new flags])],
 [AC_MSG_ERROR(m4_location[: Strange state for $1 ($acx_lib_state_[]$1). Aborting.])]
 )
 m4_ifval([$2],[_ACX_LIB_VAR_ADD([$1], [BUILD[]_ACX_LIB_IS_EXTLIB([$1],[ONLY])_CPPFLAGS], [$2])
 ])dnl
 m4_ifval([$3],[_ACX_LIB_VAR_ADD([$1], [BUILD[]_ACX_LIB_IS_EXTLIB([$1],[ONLY])_CFLAGS], [$3])
 ])dnl
 m4_ifval([$4],[_ACX_LIB_VAR_ADD([$1], [BUILD[]_ACX_LIB_IS_EXTLIB([$1],[ONLY])_CXXFLAGS], [$4])
 ])dnl
 m4_ifval([$5],[_ACX_LIB_VAR_ADD([$1], [BUILD[]_ACX_LIB_IS_EXTLIB([$1],[ONLY])_LDFLAGS], [$5])
 ])dnl
 m4_ifval([$6],[_ACX_LIB_VAR_ADD([$1], [BUILD[]_ACX_LIB_IS_EXTLIB([$1],[ONLY])_LDFLAGS_STATIC], [$6])
 ])dnl
],[m4_fatal([$1 is not declared: cannot add flags])]
)])

dnl
dnl @synopsis ACX_LIB_ADD_PUBLIC_FLAGS(VARIABLE-PREFIX, [CPPFLAGS], [CFLAGS], [CXXFLAGS], [LDFLAGS], [LDFLAGS_STATIC])
AC_DEFUN([ACX_LIB_ADD_PUBLIC_FLAGS],dnl
[_ACX_LIB_ADD_PUBLIC_FLAGS(m4_normalize([$1]),m4_normalize([$2]),
m4_normalize([$3]),m4_normalize([$4]),m4_normalize([$5]),dnl
m4_normalize([$6]))])
AC_DEFUN([_ACX_LIB_ADD_PUBLIC_FLAGS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_NEWLIB([$1], [dnl
 AS_CASE([x"$acx_lib_state_[]$1"],
 [xdefined], [dnl
   m4_ifval([$2],[dnl
     _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$2])
     _ACX_LIB_VAR_ADD([$1], [CONFIG_CPPFLAGS], [$2])
   ])dnl
   m4_ifval([$3],[dnl
     _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$3])
     _ACX_LIB_VAR_ADD([$1], [CONFIG_CFLAGS], [$3])
   ])dnl
   m4_ifval([$4],[dnl
     _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_CFLAGS], [$4])
     _ACX_LIB_VAR_ADD([$1], [CONFIG_CXXFLAGS], [$4])
   ])dnl
   m4_ifval([$5],[dnl
     _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS], [$5])
     _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS], [$5])
   ])dnl
   m4_ifval([$6],[dnl
     _ACX_LIB_VAR_ADD([$1], [PKGCONFIG_LIBS_PRIVATE], [$6])
     _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS_STATIC], [$6])
   ],[m4_ifval([$5],[dnl
     _ACX_LIB_VAR_ADD([$1], [CONFIG_LDFLAGS_STATIC], [$5])
   ])])dnl
 ],
 [xlinked], [AC_MSG_ERROR(m4_location[: the $1 library has already be linked. Cannot add new flags])],
 [x], [AC_MSG_ERROR(m4_location[: the $1 library has not been defined yet. Cannot add new flags])],
 [AC_MSG_ERROR(m4_location[: Strange state of $1 library ($acx_lib_state_[]$1). Aborting.])]
 )dnl
], [dnl
_ACX_LIB_IS_EXTLIB([$1], [dnl
 AS_CASE([x"$acx_lib_state_[]$1"],
 [xdefined], [dnl
   m4_ifval([$2],[_ACX_LIB_VAR_ADD([$1], [CPPFLAGS_PUBLIC], [$2])
   ])dnl
   m4_ifval([$3],[_ACX_LIB_VAR_ADD([$1], [CFLAGS_PUBLIC], [$3])
   ])dnl
   m4_ifval([$4],[_ACX_LIB_VAR_ADD([$1], [CXXFLAGS_PUBLIC], [$4])
   ])dnl
   m4_ifval([$5],[_ACX_LIB_VAR_ADD([$1], [LDFLAGS_PUBLIC], [$5])
   ])dnl
   m4_ifval([$5],[_ACX_LIB_VAR_ADD([$1], [LDFLAGS_STATIC_PUBLIC], [$5])
   ])dnl
 ],
 [xlinked], [AC_MSG_ERROR(m4_location[: the $1 library has already be linked. Cannot add new flags])],
 [x], [AC_MSG_ERROR(m4_location[: the $1 library has not been defined yet. Cannot add new flags])],
 [AC_MSG_ERROR(m4_location[: Strange state of $1 library ($acx_lib_state_[]$1). Aborting.])])
], [dnl
m4_fatal([$1 has not been declared with ACX_LIB][_NEW_LIB nor ACX_LIB][_CHECK_*])
])])
])
dnl ====================================================================
dnl ====================================================================
dnl Macros to cleanup flags
dnl ------------------------------
dnl
dnl
dnl @synopsis _ACX_LIB_CLEANUP_FLAGS(VARIABLE-PREFIX, VARIABLE-SUFFIX)
AC_DEFUN([_ACX_LIB_CLEANUP_FLAGS],
[dnl
  _acx_lib_cleanup_res=""
  for _acx_lib_cleanup_arg in $[]$1[]_[]$2 ; do
    AS_CASE([" $_acx_lib_cleanup_res m4_ifval([$3],[$[]$1[]_[]$3 ]) "],
      [*" $_acx_lib_cleanup_arg "*], [],
      [_acx_lib_cleanup_res="${_acx_lib_cleanup_res:+$_acx_lib_cleanup_res }$_acx_lib_cleanup_arg"]
    )    
  done
  $1[]_[]$2="$_acx_lib_cleanup_res"
])
dnl
dnl @synopsis _ACX_LIB_CLEANUP_LIBS(VARIABLE-PREFIX, VARIABLE-SUFFIX, REM, [ONLY])
AC_DEFUN([_ACX_LIB_CLEANUP_LIBS],
[dnl
  _acx_lib_cleanup_res=""
  for _acx_lib_cleanup_arg in $[]$1[]_[]$2 ; do
    AS_CASE(["$_acx_lib_cleanup_arg"],
    [-l*], [_acx_lib_cleanup_res="$_acx_lib_cleanup_arg $_acx_lib_cleanup_res"],
    [dnl
      AS_CASE([" $_acx_lib_cleanup_res m4_ifval([$3],[$[]$1[]_[]$3 ])"],
        [*" $_acx_lib_cleanup_arg "*], [],
        [_acx_lib_cleanup_res="$_acx_lib_cleanup_arg $_acx_lib_cleanup_res"]
      )dnl
    ])
  done
  $1[]_[]$2="$_acx_lib_cleanup_res"
  _acx_lib_cleanup_res=""
  for _acx_lib_cleanup_arg in $[]$1[]_[]$2 ; do
    AS_CASE(["$_acx_lib_cleanup_arg"],
    [-l*], [
      AS_CASE([" $_acx_lib_cleanup_res m4_ifval([$3],[$[]$1[]_[]$3 ])"],
        [*" $_acx_lib_cleanup_arg "*], [],
        [_acx_lib_cleanup_res="$_acx_lib_cleanup_arg${_acx_lib_cleanup_res:+ $_acx_lib_cleanup_res}"]
      )dnl
    ],
    [_acx_lib_cleanup_res="$_acx_lib_cleanup_arg${_acx_lib_cleanup_res:+ $_acx_lib_cleanup_res}"]
    )
  done
  $1[]_[]$2[]m4_ifval([$4],[_ONLY])="$_acx_lib_cleanup_res"
  m4_ifval([$4],[AC_SUBST($1[]_[]$2[]_ONLY)
    $1[]_[]$2="$[]$1[]_[]$3${_acx_lib_cleanup_res:+${[]$1[]_[]$3:+ }}$_acx_lib_cleanup_res"
  ])
])
dnl
dnl @synopsis ACX_LIB_CLEANUP_OBJ(VARIABLE-PREFIX, [noflags nolibs])
AC_DEFUN([ACX_LIB_CLEANUP_OBJ],dnl
[_ACX_LIB_CLEANUP_OBJ(m4_normalize([$1]),m4_normalize([$2]))])
AC_DEFUN([_ACX_LIB_CLEANUP_OBJ],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
_ACX_LIB_ENFORCE_NAME([$1])dnl
_ACX_LIB_IS_NEWOBJ([$1],[],[m4_fatal([$1 is not a new library or a new program])])dnl
  m4_foreach_w([_ACX_LIB_OPTION], [$2], [dnl
    m4_case(_ACX_LIB_OPTION,
    [noflags], [],
    [nolibs], [],
    [m4_fatal([unknown option ]_ACX_LIB_OPTION[ for ACX_LIB_CLEANUP])])dnl
  ])
  AS_CASE([x"$acx_lib_have_[]$1"],
  [xnew*], [AC_MSG_NOTICE([Cleaning $1 ]_ACX_LIB_NAME([$1])m4_ifval([$2],[ ($2)]))],
  [x], [AC_MSG_NOTICE([Skipping cleaning of undefined $1 ]_ACX_LIB_NAME([$1]))],
  [AC_MSG_ERROR(m4_location[: Strange kind of $1 ]_ACX_LIB_NAME([$1])[. Aborting.])]
  )dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_VAR_NEWLIBS, [dnl
    m4_bmatch(_ACX_LIB_SUFFIX,
      [INFO], [],
      [REQUIRES], [],
      [CFLAGS], [m4_if(m4_index([$2],[noflags]),[-1],
      	[_ACX_LIB_CLEANUP_FLAGS([$1], [_ACX_LIB_SUFFIX],
	  [m4_bpatsubst(_ACX_LIB_SUFFIX, [_CFLAGS], [_CPPFLAGS])])])],
      [CXXFLAGS], [m4_if(m4_index([$2],[noflags]),[-1],
      	[_ACX_LIB_CLEANUP_FLAGS([$1], [_ACX_LIB_SUFFIX],
	  [m4_bpatsubst(_ACX_LIB_SUFFIX, [_CFLAGS], [_CPPFLAGS])])])],
      [FLAGS], [m4_if(m4_index([$2],[noflags]),[-1],
      	[_ACX_LIB_CLEANUP_FLAGS([$1], [_ACX_LIB_SUFFIX])])],
      [LIBS_PRIVATE], [m4_if(m4_index([$2],[nolibs]),[-1],
        [_ACX_LIB_CLEANUP_LIBS([$1], [_ACX_LIB_SUFFIX],
      	  [m4_bpatsubst(_ACX_LIB_SUFFIX, [_PRIVATE$], [])])])],
      [LIBS_STATIC], [m4_if(m4_index($2,[nolibs]),[-1],
        [_ACX_LIB_CLEANUP_LIBS([$1], [_ACX_LIB_SUFFIX],
      	  [m4_bpatsubst(_ACX_LIB_SUFFIX, [_STATIC$], [])], [ONLY])])],
      [LIBS], [m4_if(m4_index($2,[nolibs]),[-1],
        [_ACX_LIB_CLEANUP_LIBS([$1], [_ACX_LIB_SUFFIX])])],
      [m4_fatal([unknown suffix ]_ACX_LIB_SUFFIX[ for ACX_LIB@&t@_CLEANUP])]
    )
  ])
])

dnl
dnl @synopsis ACX_LIB_CLEANUP_NEW_LIBS([noflags nolibs])
AC_DEFUN([ACX_LIB_CLEANUP_NEW_LIBS],dnl
[_ACX_LIB_CLEANUP_NEW_LIBS(m4_normalize([$1]))])
AC_DEFUN([_ACX_LIB_CLEANUP_NEW_LIBS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_LIST_NEWLIBS, [dnl
    _ACX_LIB_CLEANUP_OBJ(_ACX_LIB_SUFFIX,[$1])
  ])dnl
])
dnl
dnl @synopsis ACX_LIB_CLEANUP_NEW_PRGS([noflags nolibs])
AC_DEFUN([ACX_LIB_CLEANUP_NEW_PRGS],dnl
[_ACX_LIB_CLEANUP_NEW_PRGS(m4_normalize([$1]))])
AC_DEFUN([_ACX_LIB_CLEANUP_NEW_PRGS],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  m4_foreach_w([_ACX_LIB_SUFFIX], _ACX_LIB_LIST_NEWPRGS, [dnl
    _ACX_LIB_CLEANUP_OBJ(_ACX_LIB_SUFFIX,[$1])
  ])dnl
])
dnl
dnl @synopsis ACX_LIB_CLEANUP([noflags nolibs])
AC_DEFUN([ACX_LIB_CLEANUP],dnl
[_ACX_LIB_CLEANUP(m4_normalize([$1]))])
AC_DEFUN([_ACX_LIB_CLEANUP],
[AC_REQUIRE([_ACX_LIB_INIT])dnl
  _ACX_LIB_CLEANUP_NEW_LIBS([$1])
  _ACX_LIB_CLEANUP_NEW_PRGS([$1])
])
