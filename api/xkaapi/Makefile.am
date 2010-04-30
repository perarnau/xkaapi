##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

check_headers_SRC_C=\
	check/kaapi_hxx.cpp \
	check/kaapi_error_hxx.cpp
check_headers_SRC_CXX=\
	check/kaapi_h.c \
	check/kaapi_error_h.c
check_headers_LIBS=$(XKAAPIPROGS_BUILD_LDFLAGS) $(XKAAPIPROGS_BUILD_LIBS)

include ../check-headers.am
