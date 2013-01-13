##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

## Add poti support
BIN_PRGS=katracereader
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

katracereader_SOURCES=katracereader.cpp kaapi_trace_reader.cpp kaapi_trace_reader.h  kaapi_trace_poti.h

katracereader_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIPROGS_BUILD_CPPFLAGS)
katracereader_CXXFLAGS=$(AM_CXXFLAGS)
katracereader_LDFLAGS =$(AM_LDFLAGS)
katracereader_LDADD   =$(LIBS)

if KAAPI_USE_EXTERNAL_POTI
katracereader_CPPFLAGS+= $(POTI_CPPFLAGS)
katracereader_LDFLAGS+= $(POTI_LDFLAGS)
katracereader_LDADD+= $(POTI_LIBS)
else
katracereader_SOURCES += \
	poti/src/poti.c \
	poti/src/poti_header.c \
	poti/src/poti_events.c \
	poti/src/poti_private.h \
	poti/src/poti_config.h\
	poti/include/poti.h
katracereader_CPPFLAGS+= -I$(srcdir)/poti/include -I$(srcdir)/poti/src
endif
