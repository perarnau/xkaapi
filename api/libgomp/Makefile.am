##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(top_srcdir)/api/kaapic

EXTRA_DIST=libgomp.map libomp_ext.map libkomp-run.in

if KAAPI_LIBGOMP

pkglib_LTLIBRARIES=libgomp.la
lib_LTLIBRARIES=libkomp.la libomp_ext.la

if HAVE_VERSION_SCRIPT
libgomp_version_script = -Wl,--version-script,$(srcdir)/libgomp.map
libgomp_version_dep = $(srcdir)/libgomp.map
libomp_ext_version_script = -Wl,--version-script,$(srcdir)/libomp_ext.map
libomp_ext_version_dep = $(srcdir)/libomp_ext.map
else
libgomp_version_script =
libgomp_version_dep =
endif

libgomp_la_CPPFLAGS=$(AM_CPPFLAGS) $(LIBGOMP_BUILD_CPPFLAGS)
libgomp_la_CFLAGS=$(AM_CFLAGS) $(LIBGOMP_BUILD_CFLAGS)
libgomp_la_CXXFLAGS=$(AM_CXXFLAGS) $(LIBGOMP_BUILD_CXXFLAGS)
libgomp_la_LDFLAGS = -version-info 1:0:0 $(libgomp_version_script) \
	$(LIBGOMP_BUILD_LDFLAGS)
libgomp_la_LIBADD=../../src/libkaapi.la ../kaapic/libkaapic.la \
	$(LIBGOMP_BUILD_LIBS)
libgomp_la_DEPENDENCIES = $(libgomp_version_dep)

libkomp_la_CPPFLAGS=$(AM_CPPFLAGS) $(LIBGOMP_BUILD_CPPFLAGS) -DLIBKOMP
libkomp_la_CFLAGS=$(AM_CFLAGS) $(LIBGOMP_BUILD_CFLAGS)
libkomp_la_CXXFLAGS=$(AM_CXXFLAGS) $(LIBGOMP_BUILD_CXXFLAGS)
libkomp_la_LDFLAGS = -version-info 1:0:0 $(libgomp_version_script) $(LIBGOMP_BUILD_LDFLAGS)
libkomp_la_LIBADD=../../src/libkaapi.la ../kaapic/libkaapic.la $(LIBGOMP_BUILD_LIBS)

libomp_ext_la_CPPFLAGS=$(AM_CPPFLAGS) $(LIBGOMP_BUILD_CPPFLAGS) -DLIBKOMP
libomp_ext_la_CFLAGS=$(AM_CFLAGS) $(LIBGOMP_BUILD_CFLAGS)
libomp_ext_la_CXXFLAGS=$(AM_CXXFLAGS) $(LIBGOMP_BUILD_CXXFLAGS)
libomp_ext_la_LDFLAGS = -version-info 1:0:0 $(libomp_ext_version_script) $(LIBGOMP_BUILD_LDFLAGS)
libomp_ext_la_DEPENDENCIES = $(libomp_ext_version_dep)
#libomp_ext_la_LIBADD=-lgomp

pkglibexec_SCRIPTS = libkomp-run
CLEANFILES = $(pkglibexec_SCRIPTS)

do_subst = sed -e 's,[@]libdir[@],$(libdir),g' \
               -e 's,[@]pkglibdir[@],$(pkglibdir),g' \
               -e 's,[@]abs_top_builddir[@],$(abs_top_builddir),g' \
               -e 's,[@]PACKAGE_VERSION[@],$(PACKAGE_VERSION),g'

libkomp-run: libkomp-run.in Makefile
	$(AM_V_GEN)$(do_subst) < $(srcdir)/libkomp-run.in > $@
	$(AM_V_at)chmod +x $@

install-exec-hook: libkomp-run
	sed -e '/^# BEGIN REMOVE ONCE INSTALLED$$/,/^# END REMOVE ONCE INSTALLED$$/d' \
		> $(DESTDIR)$(pkglibexecdir)/libkomp-run \
		< libkomp-run
	chmod +x $(DESTDIR)$(pkglibexecdir)/libkomp-run
	chmod +x $(DESTDIR)$(pkglibexecdir)/libkomp-run
	$(mkdir_p) $(DESTDIR)$(bindir)
	$(LN_S) -f $(pkglibexecdir)/libkomp-run $(DESTDIR)$(bindir)/libkomp-run

include_HEADERS=

libgomp_la_SOURCES=\
    libgomp_parallel.c \
    libgomp_barrier.c \
    libgomp_critical.c \
    libgomp_ordered.c \
    libgomp_single.c \
    libgomp_lock.c \
    libgomp_loop.c \
    libgomp_loop_dynamic.c \
    libgomp_loop_static.c \
    libgomp_loop_guided.c \
    libgomp_loop_runtime.c \
    libgomp_loop_ull.c \
    libgomp_loop_wrapper_ull.c \
    libgomp_section.c \
    libgomp_task.c \
    libgomp_init.c \
    libgomp_fortran.c \
    libomprt.c\
    libgomp_g.h \
    libgomp.h \
    lock-internal.h \
    $(include_HEADERS)

libkomp_la_SOURCES=\
	$(libgomp_la_SOURCES)

libomp_ext_la_SOURCES=\
        omp_ext.h\
	libomprt_ext.c

check_PROGRAMS=check-headers

check_headers_SOURCES=\
  check-headers.c

endif
