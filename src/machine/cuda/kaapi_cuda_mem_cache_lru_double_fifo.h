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

#ifndef KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED
#define KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED

#include "kaapi_impl.h"

int kaapi_cuda_mem_cache_lru_double_fifo_init(void** data);

int kaapi_cuda_mem_cache_lru_double_fifo_insert(void* data,
                          uintptr_t ptr,
                          size_t size, kaapi_access_mode_t m);

void *kaapi_cuda_mem_cache_lru_double_fifo_remove(void* data, const size_t size);

int kaapi_cuda_mem_cache_lru_double_fifo_is_full(void* data, const size_t size);

int kaapi_cuda_mem_cache_lru_double_fifo_inc_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view,
                                                 const kaapi_access_mode_t m);

int kaapi_cuda_mem_cache_lru_double_fifo_dec_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m);

void kaapi_cuda_mem_cache_lru_double_fifo_destroy(void * proc);

#endif				/* ! KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED */
