XKAAPIDIR ?= $(HOME)/install/xkaapi_release
KASTLDIR ?= $(XKAAPIDIR)

CFLAGS	+= -I$(KASTLDIR) -I$(XKAAPIDIR)/include -DCONFIG_LIB_KASTL=1 -DNDEBUG=1
LFLAGS	+= -L$(XKAAPIDIR)/lib -lpthread -lkaapi -lnuma
