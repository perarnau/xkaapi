UCASED	:= $(shell echo $(ALGO) | tr '[:lower:]' '[:upper:]')
CFLAGS	+= -DCONFIG_ALGO_$(UCASED)=1
