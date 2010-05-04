UCASED	:= $(shell echo $(DO) | tr '[:lower:]' '[:upper:]')
CFLAGS	+= -DCONFIG_DO_$(UCASED)=1
