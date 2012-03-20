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

kaapi_event_reader_SOURCES=kaapi_event_reader.cpp kaapi_trace_reader.cpp

kaapi_event_reader_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIPROGS_BUILD_CPPFLAGS)
kaapi_event_reader_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIPROGS_BUILD_CXXFLAGS)
kaapi_event_reader_LDFLAGS =$(AM_LDFLAGS) $(KAAPIPROGS_BUILD_LDFLAGS)
kaapi_event_reader_LDADD   =$(LIBS) $(KAAPIPROGS_BUILD_LIBS)

if KAAPI_USE_EXTERNAL_POTI
kaapi_event_reader_CPPFLAGS+= $(POTI_CPPFLAGS)
kaapi_event_reader_LDFLAGS+= $(POTI_LDFLAGS)
kaapi_event_reader_LDADD+= $(POTI_LIBS)
else
kaapi_event_reader_SOURCES += \
	poti/src/poti.c \
	poti/src/poti_private.h \
	poti/src/poti_config.h\
	poti/include/poti.h
kaapi_event_reader_CPPFLAGS+= -I$(srcdir)/poti/include -I$(srcdir)/poti/src
endif
