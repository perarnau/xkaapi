
# flags to aclocal if Makefile rules trigger it
ACLOCAL_AMFLAGS=-I acinclude

# add bin for building tools when cuda dependcy resolved
SUBDIRS=src bin api network examples tests/static 

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= kaapi.pc kaapi++.pc atha.pc kastl.pc

# Add support for configure with gasnet
EXTRA_DIST=acinclude/gasnet_cppflags.mk
