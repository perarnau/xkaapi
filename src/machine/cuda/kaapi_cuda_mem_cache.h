/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
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

#ifndef KAAPI_CUDA_MEM_CACHE_H_INCLUDED
#define KAAPI_CUDA_MEM_CACHE_H_INCLUDED

#include "kaapi_impl.h"

int kaapi_cuda_mem_cache_init(kaapi_cuda_proc_t* proc);

void kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t* proc);

static inline int kaapi_cuda_mem_cache_insert(kaapi_processor_t* const kproc, uintptr_t ptr, size_t size, kaapi_access_mode_t m)
{
  return kaapi_processor_get_cudaproc(kproc)->cache.insert(kaapi_processor_get_cudaproc(kproc)->cache.data,
                                                     ptr,
                                                     size,
                                                     m);
}

static inline void* kaapi_cuda_mem_cache_remove(kaapi_processor_t* const kproc, const size_t size)
{
  return kaapi_processor_get_cudaproc(kproc)->cache.remove(kaapi_processor_get_cudaproc(kproc)->cache.data,
                                                           size);
}

static inline int kaapi_cuda_mem_cache_is_full(kaapi_processor_t* const kproc, const size_t size)
{
  return kaapi_processor_get_cudaproc(kproc)->cache.is_full(kaapi_processor_get_cudaproc(kproc)->cache.data,
                                                            size);
}

static inline int kaapi_cuda_mem_cache_inc_use(kaapi_processor_t* const kproc,
                                           uintptr_t ptr, kaapi_memory_view_t* const view,
                                           const kaapi_access_mode_t m)
{
  return kaapi_processor_get_cudaproc(kproc)->cache.inc_use(kaapi_processor_get_cudaproc(kproc)->cache.data,
                                                            ptr, view, m);
}

static inline int kaapi_cuda_mem_cache_dec_use(kaapi_processor_t* const kproc, uintptr_t ptr,
                                           kaapi_memory_view_t* const view,
                                           const kaapi_access_mode_t m)
{
  return kaapi_processor_get_cudaproc(kproc)->cache.dec_use(kaapi_processor_get_cudaproc(kproc)->cache.data,
                                                      ptr,
                                                      view,
                                                      m);
}

#endif				/* ! KAAPI_CUDA_MEM_CACHE_H_INCLUDED */
