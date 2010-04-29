XKAAPIDIR ?= $(HOME)/install
KASTLDIR ?= $(PWD)/..

SRCS	+= $(KASTLDIR)/kastl/kastl_workqueue.cpp
CFLAGS	+= -I$(KASTLDIR)/kastl -I$(XKAAPIDIR)/include -DCONFIG_LIB_KASTL=1
LFLAGS	+= -L$(HOME)/install/lib -lpthread -lxkaapi
