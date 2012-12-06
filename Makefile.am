
# flags to aclocal if Makefile rules trigger it
ACLOCAL_AMFLAGS=-I acinclude

# add bin for building tools when cuda dependcy resolved
SUBDIRS=imported-soft . src bin api examples

BUILT_SOURCES= gitlastcommit

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= kaapi.pc
if KAAPI_API_KAAPIF
pkgconfig_DATA+=kaapif.pc
endif
if KAAPI_API_KAAPIC
pkgconfig_DATA+=kaapic.pc
endif
if KAAPI_API_KAAPIXX
pkgconfig_DATA+=kaapi++.pc
endif
if KAAPI_API_KASTL
pkgconfig_DATA+=kastl.pc
endif

# Add support for configure with gasnet
EXTRA_DIST=acinclude/gasnet_cppflags.mk

# Record the last git commit when creating a tarball
EXTRA_DIST+=gitlastcommit

if GIT_REPO
clean-local:
	$(RM) $(srcdir)/gitlastcommit

.PHONY: gitlastcommit
gitlastcommit:
	@( cd $(srcdir) ; git describe --always --dirty=+ --tags --long --abbrev=16 ) > $@.new
	@if cmp --quiet $@.new "$(srcdir)/$@" ; then \
		echo "Git version not changed" ;\
		$(RM) $@.new ;\
	else \
		echo "New git version, updating $@" ;\
		mv $@.new "$(srcdir)/$@" ;\
	fi

endif
