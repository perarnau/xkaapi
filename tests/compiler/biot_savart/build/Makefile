##
## Made by fabien le mentec <texane@gmail.com>
## 

CC	:=	g++
#CC	:=	$(HOME)/install/xkaapi_rose/bin/kacc
LD	:=	$(CC)
KACC	:=	$(HOME)/install/xkaapi_rose/bin/kacc

SDL_CFLAGS :=	$(shell pkg-config --cflags sdl)
SDL_LFLAGS :=	$(shell pkg-config --libs sdl)

CFLAGS	:=	-Wall -O3 $(SDL_CFLAGS) -I../src
LFLAGS	:=	$(SDL_LFLAGS) -lm

RM	:=	rm -f

BIN	:=	bs

SRCS	:=	../src/main.cc			\
		../src/x.cc

OBJS	:=	$(SRCS:.cc=.o)


all	:	$(BIN)
$(BIN)	:	$(OBJS)
		$(LD) $(LFLAGS) -o $(BIN) $(OBJS) -L$(HOME)/install/xkaapi_rose/lib -lkaapi
clean	:
		$(RM) $(OBJS)
fclean	:	clean
		$(RM) $(BIN)
re	:	fclean all
../src/main.o:	../src/main.cc
		$(KACC) $(CFLAGS) -o $@ -c $<
%.o	:	%.cc
		$(CC) $(CFLAGS) -o $@ -c $<
