dnl @synopsis AC_CXX_LAMBDA
dnl
dnl If the compiler supports lambda expressions, define HAVE_CXX_LAMBDA.
dnl
dnl @category Cxx
dnl @author Vincent Danjean <Vincent.Danjean@ens-lyon.org>
dnl @version 2010-11-09
dnl @license AllPermissive

AC_DEFUN([AC_CXX_LAMBDA],
[AC_CACHE_CHECK(whether the compiler supports lambda expressions,
ac_cv_cxx_lambda,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[]],
    [[
auto my_lambda_func = [&](int x) { return x; };
    ]])],
 ac_cv_cxx_lambda=yes, ac_cv_cxx_lambda=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_lambda" = yes; then
  AC_DEFINE(HAVE_CXX_LAMBDA,,[define if the compiler supports lambda expressions])
fi
])

