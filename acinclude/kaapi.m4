
dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************
dnl                Kaapi libraries management
dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************


AC_DEFUN([KAAPI_PKGLIB_CHECK], [dnl
  dnl 1: library name
  dnl 2: optname
  dnl 3: NAME
  dnl 4: default (no|yes|check|)
  dnl 5: COND_NAME
  dnl 6: right-string help
  dnl 7: name
  dnl 8: pkg-config depends
  dnl 9: header.h
  dnl 10: function (to test)
  dnl 11: action-if-found
  dnl 12: action-if-not-found
  dnl 13: real-libs-to-link-with
  dnl
  USE_KAAPI_$3=false
  AC_ARG_WITH([$2],[AS_HELP_STRING([--with-$2=<instal_dir>], [$6])],[],
    [with_$2=m4_if([$4],[],[],[$4])]
  )
  AS_IF([test "$with_$2" = "no"], [
    ACX_LIB_NOLIB([$3])
  ], [
    acx_lib_have_$3=no
    m4_if([$8],[],[],[
      ACX_LIB_CHECK_PKGCONFIG([$3], ["$with_$2"], [$8])
    ])
    AS_IF([test x"$acx_lib_have_$3" = xno ],[dnl
      # Not using pkg-config
      ACX_LIB_CHECK_PLAIN([$3], ["$with_$2"], [dnl
        m4_if([$7],[],[dnl no real library to link with
	  ACX_LIB_FOUND()
	],[dnl $7 has a lib name
	  CPPFLAGS="$ACX_LIB_CPPFLAGS $ACX_LIB_LIBS"
          m4_if([$9],[],[dnl No HEADER.h file given
            AC_MSG_NOTICE([$1 library has no header file to check])
            AC_CHECK_LIB([$7], [$10],
	      [ACX_LIB_FOUND([CppFlags="$ACX_LIB_CPPFLAGS"],
                [Libs="m4_if([$13],[],[-l$7],[$13])"],
	        [LDFlags="$ACX_LIB_LIBS"])],
		[ACX_LIB_NOTFOUND], [$13])
          ],[dnl a HEADER.h file should be tested
            AC_CHECK_HEADER([$9], [
              AC_CHECK_LIB([$7], [$10],
	        [ACX_LIB_FOUND([CppFlags="$ACX_LIB_CPPFLAGS"],
                  [Libs="m4_if([$13],[],[-l$7],[$13])"],
	          [LDFlags="$ACX_LIB_LIBS"])],
		[ACX_LIB_NOTFOUND], [$13])
            ], [
	      ACX_LIB_NOTFOUND
            ])
          ])
	])dnl
      ])
    ])
    AS_IF([test "x$with_$2" != "x" && test "x$with_$2" != "xcheck"], [
      AS_IF([test $acx_lib_have_$3 = no], [
        AC_MSG_ERROR([--with-$2 was given, but $1 library cannot be found])
      ])
    ])
  ])
  AS_IF([test $acx_lib_have_$3 = yes], [
    USE_KAAPI_$3=true
    $11
  ], [
    USE_KAAPI_$3=false
    $12
  ])
  m4_if([$5],[],[],[
    AM_CONDITIONAL([$5], [test x"$USE_KAAPI_$3" = x"true"])
  ])
])

dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************
dnl                Kaapi features management
dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************

AC_DEFUN([_KAAPI_FEATURE_SET],[VAR=$1])
AC_DEFUN([_KAAPI_FEATURE_CASE_VALUE],[ [$1], [$2], ])
AC_DEFUN([_KAAPI_FEATURE_CASE_YESNO],[dnl
    FEATURE_CASE_VALUE([no,check], [AC_MSG_NOTICE([$2... no])])
    FEATURE_CASE_VALUE([no,auto],  [AC_MSG_NOTICE([$2... no])])
    FEATURE_CASE_VALUE([no,no],    [AC_MSG_NOTICE([$2... no])])
    FEATURE_CASE_VALUE([no,*],     [AC_MSG_ERROR([$2... no])])
    FEATURE_CASE_VALUE([check,*],  [AC_MSG_WARN(
	[Internal error: macro not called for --enable-$1. Please, report this bug])
	VAR=no
	AC_MSG_NOTICE([$2... disabled])])
    FEATURE_CASE_VALUE([yes,*],    [AC_MSG_NOTICE([$2... yes])])
])
AC_DEFUN([KAAPI_FEATURE], [dnl
  dnl 1: name
  dnl 2: right-string help
  dnl 3: description
  dnl 4: default (yes/no/check)
  dnl 5: action-if-enabled
  dnl 6: action-if-not-enabled
  dnl 7: COND_NAME
  dnl 8: check_code
  dnl 9: verif results
  dnl
  m4_pushdef([VAR],[m4_translit([enable_$1], [-+.], [___])])
  AC_ARG_ENABLE([$1],
    [AS_HELP_STRING([--enable-$1], [$2 (default $4)])],
    [], [VAR=$4])

  VAR[]_user="$VAR"

  if test x"$VAR" != x"no"; then
    m4_pushdef([FEATURE_ENABLE],[VAR=yes])
    m4_pushdef([FEATURE_SET],m4_defn([_KAAPI_FEATURE_SET]))
    m4_pushdef([FEATURE_DISABLE],[VAR=no])
    m4_if([$8],[],[ENABLE_FEATURE],[
      # Checking if $1 must be enabled
      $8
    ])
    m4_popdef([FEATURE_ENABLE])
    m4_popdef([FEATURE_SET])
    m4_popdef([FEATURE_DISABLE])
  fi

  m4_pushdef([FEATURE_CASE_VALUE], m4_defn([_KAAPI_FEATURE_CASE_VALUE]))
  m4_pushdef([FEATURE_CASE_YESNO], [_KAAPI_FEATURE_CASE_YESNO([$1],[$3])])
  AS_CASE(["$VAR,$VAR[]_user"],
    m4_if([$9],[],[FEATURE_CASE_YESNO],[$9])
    [   AC_MSG_WARN([strange argument $VAR for --enable-$1])
	VAR=no
	AC_MSG_NOTICE([$3... disabled])]
  )
  m4_popdef([FEATURE_CASE_YESNO])
  m4_popdef([FEATURE_CASE_VALUE])

  if test x"$VAR" != x"no"; then :
    $5
  else :
    $6
  fi
  m4_if([$7],[],[],[
    AM_CONDITIONAL([$7], [test x"$VAR" != x"no"])
  ])
  FEATURE_RESUME="${FEATURE_RESUME}
  $2    => $VAR"
  m4_popdef([VAR])
])

AC_DEFUN([KAAPI_FEATURE_RESUME], [dnl
  AC_MSG_NOTICE([Selected features in this release:$FEATURE_RESUME])
])

dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************
dnl                Kaapi documentation management
dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************

AC_DEFUN([KAAPI_ARG_DOC_SETVARS], [dnl
  [kaapi_]AS_TR_SH([$1])[_build]=$2
  [kaapi_]AS_TR_SH([$1])[_install]=$3
])

AC_DEFUN([KAAPI_ARG_DOC], [dnl
    dnl 1: tag
    dnl 2: second help string
    dnl 3: depending option
    dnl 4: with
    dnl 5: without
    AC_ARG_WITH([$1],
      [AS_HELP_STRING([--with-$1@<:@=no|try|yes|force-build|install-only@:>@], [$2])],
      [ kaapi_doc_default=no
        AS_CASE([$with_[]AS_TR_SH($1)],
	  [no], [dnl
	    KAAPI_ARG_DOC_SETVARS([$1], [no], [no])
        ],[try], [dnl
	    KAAPI_ARG_DOC_SETVARS([$1], [try], [try])
	],[yes], [dnl
	    KAAPI_ARG_DOC_SETVARS([$1], [try], [yes])
        ],[force-build], [dnl
	    KAAPI_ARG_DOC_SETVARS([$1], [yes], [yes])
        ],[install-only], [dnl
	    KAAPI_ARG_DOC_SETVARS([$1], [no], [yes])
        ],[dnl
            AC_MSG_WARN([Ignoring unkown value $with_doc for option --with-$1])
	    kaapi_doc_default=yes
        ])
      ], [
        kaapi_doc_default=yes
      ])
    AS_IF([ test $kaapi_doc_default = yes ], [dnl
	KAAPI_ARG_DOC_SETVARS([$1],
	    [$kaapi_[]AS_TR_SH([doc]m4_if([$3],[],[],[-$3]))_build],
	    [$kaapi_[]AS_TR_SH([doc]m4_if([$3],[],[],[-$3]))_install])
        $5
      ], [$4])
  ])

AC_DEFUN([KAAPI_DOC_INIT], [dnl
    dnl check if we want to generate documentation
    AC_MSG_NOTICE([checking weather KAAPI documentation can be built and/or installed])

    KAAPI_ARG_DOC([doc], [build documentation [default=try]], [],
      [], [dnl
        kaapi_doc_build=try
        kaapi_doc_install=try        
      ])

    AC_MSG_CHECKING([weather source and build trees are merged])
    kaapi_doc_srcdir="$( test x"$srcdir" != x && cd "$srcdir" ; /bin/pwd )"
    kaapi_doc_builddir="$( test x"$builddir" != x && cd "$builddir" ; /bin/pwd )"
    if test x"$kaapi_doc_srcdir" = x"$kaapi_doc_builddir" ; then
      kaapi_doc_onplace=yes
    else
      kaapi_doc_onplace=no
    fi
    AC_MSG_RESULT([$kaapi_doc_onplace])
  ])

AC_DEFUN([KAAPI_DOC_GROUP], [dnl
    dnl 1: tag
    dnl 2: Help string
    dnl 3: depending option

    AC_REQUIRE([KAAPI_DOC_INIT])
    
    KAAPI_ARG_DOC([doc-$1],
      [build $2 @<:@default: same as --with-doc]m4_if([$3],[],
        [],[-$3])[@:>@], [$3],
      [], [])
  ])

AC_DEFUN([KAAPI_DOC], [dnl
    dnl 1: tag
    dnl 2: Help string
    dnl 3: depending option
    dnl 4: files to check
    dnl 5: Progs to check
#TODO
    dnl tools here AND (doc not here OR build_dir == src_dir) => build possible
    dnl build possible AND build requested => build enabled
    dnl doc here or build enabled => intallation of doc possible
    dnl intallation of doc possible AND not disabled => intallation of doc
    dnl build enabled => clean enabled
    
    KAAPI_DOC_GROUP([$1],[$2],[$3])

    AS_IF([test "$kaapi_doc_[]AS_TR_SH([$1])_build" = no],
      [ kaapi_doc_check_depends=no
        kaapi_doc_check_reason="disabled by user"
      ],[
        AC_MSG_CHECKING([missing program(s) to build $2])
        kaapi_doc_check_depends=yes
        kaapi_doc_check_missing=none
        m4_foreach_w([prog], [$5], [
          AC_REQUIRE([PROG_]prog)
          if test $have_[]m4_tolower(prog) != yes ; then
	      kaapi_doc_check_depends=no
	      if test $kaapi_doc_check_missing = none ; then
	          kaapi_doc_check_missing="m4_tolower(prog)"
	      else
	          kaapi_doc_check_missing="$kaapi_doc_check_missing, m4_tolower(prog)"
	      fi
          fi
        ])
	AC_MSG_RESULT([$kaapi_doc_check_missing])
        kaapi_doc_check_reason="missing $kaapi_doc_check_missing"
      ])

    # Is doc already built in sources ?
    kaapi_doc_check_present=no
    m4_foreach_w([filename], [$4], [
        if test -f "$srcdir/filename" ; then
	    kaapi_doc_check_present=yes
        fi
    ])
    
    AC_MSG_CHECKING([weather $2 will be built])
    kaapi_doc_check_nodoublebuild=no
    if test $kaapi_doc_check_depends = no ; then
      # Some depends are missing
      kaapi_doc_check_build=no
    else
      if test $kaapi_doc_check_present = no \
         || test x"$kaapi_doc_onplace" = xyes ; then
	# if no doc is present or if the doc can be built in the tree
        kaapi_doc_check_build=yes
      else
        # no way to rebuild doc in external build tree
        kaapi_doc_check_build=no
	kaapi_doc_check_nodoublebuild=yes
        kaapi_doc_check_reason="already built in source tree"
      fi
    fi
    AS_IF([test $kaapi_doc_check_build = yes],
      [ AC_MSG_RESULT([yes])],
      [ AC_MSG_RESULT([no ($kaapi_doc_check_reason)]) ])

    AS_IF(
      [ test $kaapi_doc_check_build = no \
        && test $kaapi_doc_[]AS_TR_SH([$1])_build != no ],
      [
        AS_IF(
  	  [ test $kaapi_doc_check_nodoublebuild = yes ],
	  [ AC_MSG_NOTICE([Cannot rebuild $2 when already present in the source tree that is different from the build tree])
	    AC_MSG_NOTICE([Try './configure --with-doc-$1 ; make distclean' in the source tree to remove pre-built documentation])
	  ])

        AS_IF([test $kaapi_doc_[]AS_TR_SH([$1])_build = yes],
	  [
	    AC_MSG_ERROR([Unable to build $2 as requested])
	  ])
      ])

    AC_MSG_CHECKING([weather $2 will be installed])
    kaapi_doc_check_install_msg=""
    AS_IF(
      [ test $kaapi_doc_check_build = yes],
        [ kaapi_doc_check_can_install=yes
        ],
      [ test $kaapi_doc_check_present = yes ],
        [ kaapi_doc_check_can_install=yes
          kaapi_doc_check_install_msg=" (from prebuilt files)"
        ],
      [ kaapi_doc_check_can_install=no
      ])
    AS_IF(
      [ test $kaapi_doc_[]AS_TR_SH([$1])_install = no ],
      [
        kaapi_doc_check_install=no
	kaapi_doc_check_install_msg=" (disabled by user)"
      ],
      [ test $kaapi_doc_check_can_install = no ],
      [
        kaapi_doc_check_install=no
      ],
      [
        kaapi_doc_check_install=yes
      ])
    AC_MSG_RESULT([$kaapi_doc_check_install$kaapi_doc_check_install_msg])
    AS_IF(
      [ test $kaapi_doc_check_install = no \
        && test $kaapi_doc_[]AS_TR_SH([$1])_install = yes ],
      [
	AC_MSG_ERROR([Unable to install $2 as requested])
      ])

    kaapi_doc_[]AS_TR_SH([$1])_build=$kaapi_doc_check_build
    kaapi_doc_[]AS_TR_SH([$1])_install=$kaapi_doc_check_install

    AM_CONDITIONAL([KAAPI_DOC_]m4_toupper(AS_TR_SH([$1]))[_BUILD],
      [ test $kaapi_doc_[]AS_TR_SH([$1])_build = yes ])
dnl For now, documentation is build and clean under the same conditions
    AM_CONDITIONAL([KAAPI_DOC_]m4_toupper(AS_TR_SH([$1]))[_CLEAN],
      [ test $kaapi_doc_[]AS_TR_SH([$1])_build = yes ])
    AM_CONDITIONAL([KAAPI_DOC_]m4_toupper(AS_TR_SH([$1]))[_INSTALL],
      [ test $kaapi_doc_[]AS_TR_SH([$1])_install = yes ])

])

dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************
dnl                Kaapi flags management
dnl ******************************************************************
dnl ******************************************************************
dnl ******************************************************************

AC_DEFUN([KAAPI_CHECK_COMPILER_FLAGS], [dnl
dnl 1: FLAGS
dnl 2: compiler-option
dnl
dnl if $FLAGS is not set on the command line, this macro adds
dnl the compiler option if supported
    if test x"$ac_test_$1" != xset; then
         AX_CHECK_COMPILER_FLAGS([$2],[$1="$[]$1 $2"])
    fi
])
