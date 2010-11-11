##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@
AM_CFLAGS=@AM_CFLAGS@
AM_CXXFLAGS=@AM_CXXFLAGS@


check_PROGRAMS=

include accumulate.am
include cilk.am
include fibo.am
include find.am
include for_each.am
include hello.am
include nqueens.am
include poisson3D.am
include branchbound.am

USER_MAKEFILES=\
	accumulate/Makefile \
	cilk/Makefile \
	fibo/Makefile \
	find/Makefile \
	for_each/Makefile \
	hello/Makefile \
	nqueens/Makefile \
	poisson3D/Makefile \
	branchbound/Makefile

TESTS=  

EXTRA_DIST=gen-Makefile.m4

###########################################
# standalone Makefiles generation

pkgconfigdir = $(libdir)/pkgconfig

%/Makefile: gen-Makefile.m4 gen-%.am.m4
	m4 $^ > $@
	echo "###################" >> $@
	echo "PKG_CONFIG=env PKG_CONFIG_PATH=$(pkgconfigdir) pkg-config" >> $@

SEP=
S=$(SEP) 	$(SEP)
gen-%.am.m4: %.am Makefile
	echo 'define([DIR],[$*])dnl' > $@
	sed < $< -e :a -e '/\\$$/N; s/\\\n//; ta' \
	  | sed -e '/^[$S]*#/d' \
	  | sed -e '/^[^=]\+_\(SOURCES\|PRGS\|\(C\(PP\|XX\|\)\|LD\)FLAGS\|LDADD\)[$S]*=/p;d' \
	  > $@.temp
	sed < $@.temp -e 's/^[$S]*\([^$S][^=]*_PRGS\)[$S]*=[$S]*\(.*\)$$/\1/p;d' \
	  | while read VAR ; do \
	    VALUE="$$($(MAKE) -s VAR="$$VAR" .get-var)" ; \
	    echo "define([PROGS],[$$VALUE])dnl" ; \
	  done >> $@
	sed < $@.temp -e '/^[$S]*\([^$S][^=]*_SOURCES\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | sed -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE) -s VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	sed < $@.temp -e '/^[$S]*\([^$S][^=]*_[^_]\+FLAGS\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | sed -e 's/$$(\(AM\|[A-Z]*_BUILD\)_[^_]\+FLAGS)//g' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | sed -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE) -s VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	sed < $@.temp -e '/^[$S]*\([^$S][^=]*_LDADD\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | sed -e 's/$$(\(\|[A-Z]*_BUILD_\)LIBS)//g' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | sed -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE) -s VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	sed < $@.temp -e '/^[$S]*\([^$S][^=]*_LDADD\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | sed -e 's/_LDADD[$S]*=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="" ; \
	    case " $$EQ " in *'$$(KAAPIPROGS_BUILD_LIBS)'*) VALUE="$$VALUE kaapi" ;; esac ; \
	    case " $$EQ " in *'$$(KAAPIXXPROGS_BUILD_LIBS)'*) VALUE="$$VALUE kaapi++" ;; esac ; \
	    case " $$EQ " in *'$$(ATHAPROGS_BUILD_LIBS)'*) VALUE="$$VALUE atha" ;; esac ; \
	    echo "define([$${VAR}_PKGLIBS],[$$VALUE])dnl" ; \
	  done >> $@
	echo 'Makefile' >> $@
	$(RM) $@.temp

clean-local:
	$(RM) gen-*.am.m4 */Makefile

.get-var:
	@echo '$($(VAR))'
###########################################


###########################################
# examples installation

examplesdir=$(docdir)/examples

noinst_DATA=$(USER_MAKEFILES)

install-data-hook: install-data-hook-examples
uninstall-hook: uninstall-hook-examples

.PHONY: install-data-hook-examples uninstall-hook-examples
install-data-hook-examples: $(DIST_SOURCES)
	set -e ; \
	for f in $(DIST_SOURCES) ; do \
		$(MKDIR_P) "$(DESTDIR)$(examplesdir)/$$(dirname $$f)" ;\
	        $(install_sh_DATA) "$(srcdir)/$$f" "$(DESTDIR)$(examplesdir)/$$(dirname $$f)/" ; \
	done ;
	for f in $(USER_MAKEFILES) ; do \
		$(MKDIR_P) "$(DESTDIR)$(examplesdir)/$$(dirname $$f)" ;\
	        $(install_sh_DATA) "$$f" "$(DESTDIR)$(examplesdir)/$$(dirname $$f)/" ; \
	done ;

uninstall-hook-examples:
	for makefile in $(USER_MAKEFILES); do\
		dir="$$(dirname $$makefile)" ; \
		$(RM) -r $(DESTDIR)$(examplesdir)/$$dir ;\
	done

###########################################

