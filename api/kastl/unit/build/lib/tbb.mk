TBBDIR	?= $(HOME)/install

CFLAGS	+= -I$(TBBDIR)/include -DCONFIG_LIB_TBB=1
LFLAGS	+= -L$(TBBDIR)/lib -lpthread -ltbb
