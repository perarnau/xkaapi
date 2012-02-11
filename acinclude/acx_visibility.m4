dnl
dnl 
AC_DEFUN([ACX_VISIBILITY], [

AC_CACHE_CHECK([for __attribute__((visibility("hidden")))],
    acx_visibility_cv_hidden_attribute, [
        AC_TRY_LINK([], 
                    [extern __attribute__ ((visibility ("hidden"))) int foo;],
                    [acx_visibility_cv_hidden_attribute=yes],
                    [acx_visibility_cv_hidden_attribute=no])])
AC_CACHE_CHECK([for __attribute__((visibility("internal")))],
    acx_visibility_cv_internal_attribute, [
        AC_TRY_LINK([], 
                    [extern __attribute__ ((visibility ("internal"))) int foo;],
                    [acx_visibility_cv_internal_attribute=yes],
                    [acx_visibility_cv_internal_attribute=no])])

if test $acx_visibility_cv_hidden_attribute = yes; then
  AC_DEFINE(HAVE_VISIBILITY_HIDDEN,, Define if __attribute__ ((visibility ("hidden"))) present)
fi
if test $acx_visibility_cv_internal_attribute = yes; then
  AC_DEFINE(HAVE_VISIBILITY_INTERNAL,, Define if __attribute__ ((visibility ("internal"))) present)
fi

])

