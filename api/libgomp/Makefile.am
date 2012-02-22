##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(top_srcdir)/api/kaapic

if KAAPI_LIBGOMP

SUBDIRS=tests

pkglib_LTLIBRARIES=libgomp.la
lib_LTLIBRARIES=libkomp.la

libgomp_version_script = -Wl,--version-script,$(srcdir)/libgomp.map
libgomp_version_dep = $(srcdir)/libgomp.map

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
libkomp_la_LDFLAGS = -version-info 1:0:0 $(LIBGOMP_BUILD_LDFLAGS)
libkomp_la_LIBADD=../../src/libkaapi.la ../kaapic/libkaapic.la $(LIBGOMP_BUILD_LIBS)

bin_SCRIPTS = kaapi-gomp-wrapper
CLEANFILES = $(bin_SCRIPTS)
EXTRA_DIST = kaapi-gomp-wrapper.in

do_subst = sed -e 's,[@]libdir[@],$(libdir),g' \
               -e 's,[@]pkglibdir[@],$(pkglibdir),g' \
               -e 's,[@]abs_top_builddir[@],$(abs_top_builddir),g'

kaapi-gomp-wrapper: kaapi-gomp-wrapper.in Makefile
	$(do_subst) < $(srcdir)/kaapi-gomp-wrapper.in > $@
	chmod +x $@

install-exec-hook: kaapi-gomp-wrapper
	sed -e 's/^wrapper_installed=.*/wrapper_installed=yes/' \
		> $(DESTDIR)$(bindir)/kaapi-gomp-wrapper \
		< kaapi-gomp-wrapper
	chmod +x $(DESTDIR)$(bindir)/kaapi-gomp-wrapper

include_HEADERS=\
    libgomp.h

libgomp_la_SOURCES=\
    libgomp_parallel.c \
    libgomp_barrier.c \
    libgomp_critical.c \
    libgomp_ordered.c \
    libgomp_single.c \
    libgomp_loop.c \
    libgomp_loop_dynamic.c \
    libgomp_loop_runtime.c \
    libgomp_loop_static.c \
    libgomp_section.c \
    libgomp_task.c \
    libgomp_init.c \
    libomprt.c\
    $(include_HEADERS)

libkomp_la_SOURCES=\
	$(libgomp_la_SOURCES)

endif
