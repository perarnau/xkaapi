/*
** kaapi_defs.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#ifndef _KAAPI_DEFS_H
#define _KAAPI_DEFS_H 1

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef __linux__
#  ifdef HAVE_UCONTEXT_H
#    define KAAPI_USE_UCONTEXT
#  elif HAVE_SETJMP_H
#    error "Not implemented yet"
#    define KAAPI_USE_SETJMP
#  endif
#endif

#ifdef __APPLE__
#  include <libkern/OSAtomic.h>
#endif

#ifdef __APPLE__
#  ifdef HAVE_SETJMP_H
#    define KAAPI_USE_SETJMP
#  elif HAVE_UCONTEXT_H
#    define KAAPI_USE_UCONTEXT
#  endif
#endif

#if not (defined(__i386__) || defined(__x86_64__) || defined(__ia64__) || \
	 defined(__PPC__) || defined(__PPC64__) || defined(__arm__))
#  error "Unsupported Architecture"
#endif

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_DEFS_H */
