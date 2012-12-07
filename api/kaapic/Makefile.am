##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_API_KAAPIC

lib_LTLIBRARIES=libkaapic.la

libkaapic_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICLIB_BUILD_CPPFLAGS)
libkaapic_la_CFLAGS=$(AM_CFLAGS) $(KAAPICLIB_BUILD_CFLAGS)
libkaapic_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPICLIB_BUILD_CXXFLAGS)
libkaapic_la_LDFLAGS = -version-info 1:0:1 $(KAAPICLIB_BUILD_LDFLAGS)
libkaapic_la_LIBADD=../../src/libkaapi.la $(KAAPICLIB_BUILD_LIBS)

M4_BUILT_FILES=\
  kaapic_dfg_switch.h \
  kaapic_adaptive_switch.h \
  kaapic_adaptive_switch_ull.h

M4_SRCS=\
  kaapic_dfg_switch.m4 \
  kaapic_adaptive_switch.m4 \
  kaapic_adaptive_switch_ull.m4 \
  m4_generator \
  m4_macros

BUILT_SOURCES = $(M4_BUILT_FILES) $(srcdir)/../kaapif/kaapif_inc.h
EXTRA_DIST = $(M4_SRCS)

$(M4_BUILT_FILES): stamp-m4_generator

# We do not track here dependencies of kaapif_inc.h itself
# We will need to build the project two times if
# - this is not the first build (ie kaapif_inc.h already exists)
# - and a dependency of kaapif_inc.h changed
$(srcdir)/../kaapif/kaapif_inc.h:
	$(MAKE) -C ../kaapif kaapif_inc.h

.INTERMEDIATE: stamp-m4_generator
stamp-m4_generator: $(M4_SRCS)
	cd $(srcdir) && ./m4_generator 32
	touch $@

clean-local:
	$(RM) stamp-m4_generator

include_HEADERS=\
	kaapic.h

libkaapic_la_SOURCES=\
	kaapic.c \
	kaapic_impl.h \
	kaapic_spawn.c \
	kaapic_spawn_attr.c \
	kaapic_alloca.c \
	kaapic_work.c \
	kaapic_work_ull.c \
	kaapic_foreach_docall.c \
	kaapic_foreach_docall_ull.c \
	kaapic_foreach_attr.c \
	kaapic_foreach.c \
	kaapic_foreach_ull.c \
	kaapic_foreach_withformat.c \
	kaapic_foreach_withformat_ull.c \
	$(include_HEADERS)\
	kaapic_impl.h \
	$(M4_BUILT_FILES)

endif
