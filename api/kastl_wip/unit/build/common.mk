PWD	:= $(shell pwd)
SRCDIR	:= $(PWD)/..

CC	:= g++
LD	:= g++
CFLAGS	:= -std=gnu++0x -Wall -O3
LFLAGS	:=
SRCS	:= $(SRCDIR)/main.cc
OBJS	:= $(SRCS:.cc=.o)

LIB	?= none
DO	?= none
ALGO	?= none
