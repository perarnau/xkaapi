##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

BIN_PRGS=kaapi_event_reader
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kaapi_event_reader_SOURCES=kaapi_event_reader.cpp PajeGenerator.cpp PajeGenerator.hpp

kaapi_event_reader_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIPROGS_BUILD_CPPFLAGS)
kaapi_event_reader_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIPROGS_BUILD_CXXFLAGS)
kaapi_event_reader_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIPROGS_BUILD_LDFLAGS)
kaapi_event_reader_LDADD   =$(LIBS)        $(KAAPIPROGS_BUILD_LIBS)
