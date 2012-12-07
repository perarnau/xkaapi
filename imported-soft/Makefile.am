
SUBDIRS=

SUBDIRS+=libffi

install-dvi-recursive install-info-recursive: install-%-recursive:
	@: skip recursive $*

dvi-recursive info-recursive install-recursive: %-recursive:
	@: skip recursive $*

check-recursive:
	@: skip recursive check

