XKAAPIDIR ?= $(HOME)/install
#KASTLDIR ?= $(PWD)/../..
KASTLDIR ?= $(XKAAPIDIR)/include

CFLAGS	+= -I$(KASTLDIR) -I$(XKAAPIDIR)/include -DCONFIG_LIB_KASTL=1 -DNDEBUG=1
LFLAGS	+= -L$(HOME)/install/lib -lpthread -lxkaapi -lnuma -lcuda
