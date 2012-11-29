##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@
AM_CFLAGS=@AM_CFLAGS@
AM_CXXFLAGS=@AM_CXXFLAGS@


check_PROGRAMS=

#include accumulate.am
include array.am
#include bfs.am
include branchbound.am
include cilk.am
include fibo.am
#include find.am
include for_each.am
include hello.am
#include jacobi2D.am
include kaapic.am
#include linearwork.am
include listranking.am
include matrix.am
include mergesort.am
include net.am
include nqueens.am
include poisson3D.am
#include prefix.am
include papi.am
include cuda.am

USER_MAKEFILES=\
	array/Makefile \
	branchbound/Makefile \
	cilk/Makefile \
	fibo/Makefile \
	for_each/Makefile \
	kaapic/Makefile \
	listranking/Makefile \
	hello/Makefile \
	matrix/Makefile \
	mergesort/Makefile \
	net/Makefile \
	nqueens/Makefile \
	poisson3D/Makefile \
	papi/Makefile \
	cuda/Makefile 

#TO CORRECT	\
	accumulate/Makefile\
	bfs/Makefile \
	find/Makefile \
	jacobi2D/Makefile \
    linearwork/Makefile \
	prefix/Makefile


TESTS=  

installcheck-local:
	set -e ; \
	for f in $(USER_MAKEFILES) ; do \
		d="$$(dirname "$$f")" ; \
		$(MAKE) $(AM_MAKEFLAGS) -C "$(examplesdir)/$$d" ; \
	done

EXTRA_DIST=gen-Makefile.m4

###########################################
# standalone Makefiles generation

pkgconfigdir = $(libdir)/pkgconfig

%/Makefile: gen-Makefile.m4 gen-%.am.m4
	$(AM_V_GEN)m4 $^ \
		| $(SED) -e "s|^PKGCONFIG=$$|PKG_CONFIG=env PKG_CONFIG_PATH=$(pkgconfigdir) pkg-config|" \
		| $(SED) -e "s|^CC=$$|CC=$(CC)|" \
		| $(SED) -e "s|^CXX=$$|CXX=$(CXX)|" > $@

SEP=
S=$(SEP) 	$(SEP)
gen-%.am.m4.info: %.am
	$(AM_V_GEN)$(SED) < $< -e :a -e '/\\$$/N; s/\\\n//; ta' \
	  | $(SED) -e '/^[$S]*#/d' \
	  | $(ESED) -e '/^[^=]+_(SOURCES|PRGS|(C(PP|XX)?|LD)FLAGS|LDADD)[$S]*=/p;d' \
	  > $@

MAKE_VAR=$(MAKE) -s --no-print-directory
gen-%.am.m4: gen-%.am.m4.info Makefile
	$(AM_V_GEN)echo 'define([DIR],[$*])dnl' > $@
	+$(AM_V_at)$(SED) < $< -e \
		's/^[$S]*\([^$S][^=]*_PRGS\)[$S]*=[$S]*\(.*\)$$/\1/p;d' \
	  | while read VAR ; do \
	    VALUE="$$($(MAKE_VAR) VAR="$$VAR" .get-var)" ; \
	    echo "define([PROGS],[$$VALUE])dnl" ; \
	  done >> $@
	+$(AM_V_at)$(SED) < $< -e \
		'/^[$S]*\([^$S][^=]*_SOURCES\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | $(SED) -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE_VAR) VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	+$(AM_V_at)$(ESED) < $< -e \
		'/^[$S]*\([^$S][^=]*_[^_]+FLAGS\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | $(ESED) -e 's/$$\((AM|[A-Z]*_BUILD)_[^_]+FLAGS\)//g' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | $(SED) -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE_VAR) VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	+$(AM_V_at)$(SED) < $< -e \
		'/^[$S]*\([^$S][^=]*_LDADD\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | $(ESED) -e 's/$$\(([A-Z]*_BUILD_)?LIBS\)//g' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | $(SED) -e 's/=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="$$($(MAKE_VAR) VAR="$$VAR" "$$EQ" .get-var)" ; \
	    echo "define([$$VAR],[$$VALUE])dnl" ; \
	  done >> $@
	$(AM_V_at)$(SED) < $< -e \
		'/^[$S]*\([^$S][^=]*_LDADD\)[$S]*=[$S]*\(.*\)$$/p;d' \
	  | while read EQ ; do \
	    VAR=$$(echo "$$EQ" | $(SED) -e 's/_LDADD[$S]*=.*//' -e 's/[$S]//g' ) ;\
	    VALUE="" ; \
	    case " $$EQ " in *'$$(KAAPIPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE kaapi" ;; esac ; \
	    case " $$EQ " in *'$$(KAAPIXXPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE kaapi++" ;; esac ; \
	    case " $$EQ " in *'$$(KAAPICPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE kaapic" ;; esac ; \
	    case " $$EQ " in *'$$(KAAPIFPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE kaapif" ;; esac ; \
	    case " $$EQ " in *'$$(ATHAPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE atha" ;; esac ; \
	    case " $$EQ " in *'$$(KASTLPROGS_BUILD_LIBS)'*) \
		VALUE="$$VALUE kastl" ;; esac ; \
	    echo "define([$${VAR}_PKGLIBS],[$$VALUE])dnl" ; \
	  done >> $@
	$(AM_V_at)echo 'Makefile' >> $@

clean-local:
	$(RM) gen-*.am.m4 gen-*.am.m4.info */Makefile

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
