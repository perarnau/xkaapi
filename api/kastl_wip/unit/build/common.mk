PWD	:= $(shell pwd)
SRCDIR	:= $(PWD)/../src
BINDIR	:= $(PWD)/../bin

CC	:= g++
LD	:= g++
CFLAGS	:= -std=gnu++0x -Wall -O3
LFLAGS	:=
SRCS	:= $(SRCDIR)/main.cpp

LIB	?= none
DO	?= none
ALGO	?= none
