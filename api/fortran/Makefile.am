##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_TARGET_FORTRAN

lib_LTLIBRARIES=libkaapif.la

libkaapif_la_CPPFLAGS=$(AM_CPPFLAGS) $(KFORTRANLIB_BUILD_CPPFLAGS) 
libkaapif_la_CFLAGS=$(AM_CFLAGS) $(KFORTRANLIB_BUILD_CFLAGS) $(STD_C99_FLAGS)
libkaapif_la_CXXFLAGS=$(AM_CXXFLAGS) $(KFORTRANLIB_BUILD_CXXFLAGS) $(STD_FORTRAN_FLAGS)
libkaapif_la_LDFLAGS = -version-info 1:0:1 $(KFORTRANLIB_BUILD_LDFLAGS)
libkaapif_la_LIBADD=../../src/libkaapi.la $(KFORTRANLIB_BUILD_LIBS)


FORTRAN_H=\
kaapif.h

libkaapif_la_SOURCES=\
kaf_init.c\
kaf_getargs.f\
kaf_task.c\
${FORTRAN_H}

endif
