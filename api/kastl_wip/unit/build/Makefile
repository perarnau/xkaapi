# makefile variable
# DO={bench,check}
# LIB={tbb,pastl,kastl,stl}
# ALGO={for_each,min_element,...}


include common.mk

ifneq ($(LIB),none)
	-include lib/common.mk
	-include lib/$(LIB).mk
endif

ifneq ($(DO),none)
	-include do/common.mk
	-include do/$(DO).mk
endif

ifneq ($(ALGO),none)
	-include algo/common.mk
	-include algo/$(ALGO).mk
endif

OBJS	:= $(SRCS:.cpp=.o)

BIN	:= $(BINDIR)/$(ALGO)-$(LIB)-$(DO)

all	: $(BIN)
none-*	:
*-none-*:
	echo FOOBAR 
$(BIN)	: clean $(OBJS)
	$(LD) $(LFLAGS) -o $(BIN) $(OBJS)
%.o	: %.cpp
	$(CC) $(CFLAGS) -c -o $(@) $(<)
%.o	: %.c
	$(CC) $(CFLAGS) -c -o $(@) $(<)
clean	:
	-rm $(OBJS) 2> /dev/null
fclean	: clean
	-rm -rf $(BINDIR) 2> /dev/null
