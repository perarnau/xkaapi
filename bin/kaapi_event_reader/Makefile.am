##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

## Add poti support
BIN_PRGS=kaapi_event_reader
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kaapi_event_reader_SOURCES=kaapi_event_reader.cpp \
	poti/src/poti.c \
	poti/src/poti_private.h \
	poti/include/poti_config.h\
	poti/include/poti.h

kaapi_event_reader_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIPROGS_BUILD_CPPFLAGS) -I$(srcdir)/poti/include -I$(srcdir)/poti/src
kaapi_event_reader_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIPROGS_BUILD_CXXFLAGS)
kaapi_event_reader_LDFLAGS =$(AM_LDFLAGS)
kaapi_event_reader_LDADD   =$(LIBS)

