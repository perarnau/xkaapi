This is the XKAAPI library. 

===========================================================================
Find information at http://kaapi.gforge.inria.fr/
===========================================================================

===========================================================================
If you got the source files and want to compile and install the library,
read the INSTALL file
===========================================================================

== About XKAAPI ==

XKAAPI means C implementation of Kernel for Adaptative, Asynchronous Parallel
and Interactive programming. It is a C library that allows to execute
multithreaded computation with data flow synchronization between threads. The
library is able to schedule fine/medium size grain program on distributed
machine. The data flow graph is dynamic (unfold at runtime).
Target architectures are **clusters of SMP machines** and MPSOC.

== Main features are ==
 - It is based on work-stealing algorithms ;
 - It can run on various processors ;
 - It can run on various architectures (clusters or grids) ;
 - It contains non-blocking and scalable algorithms ;
 - It is based on a two-level thread (user and system threads);

