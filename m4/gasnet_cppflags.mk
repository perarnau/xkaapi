

include ${GASNET_CONDUIT_MAK}

all:
	export GASNET_CPPFLAGS="${GASNET_CPPFLAGS}"
	export GASNET_CFLAGS="${GASNET_CFLAGS}"
	export GASNET_LDFLAGS="${GASNET_LDFLAGS}"
	export GASNET_LIBS="${GASNET_LIBS}"

