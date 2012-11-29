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

#include "kaapi_impl.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache.h"
#include "kaapi_cuda_mem_cache_lru_fifo.h"
#include "kaapi_cuda_mem_cache_lru_double_fifo.h"

static inline void kaapi_cuda_mem_cache_lru_fifo_config(kaapi_cuda_proc_t* proc)
{
  proc->cache.init = &kaapi_cuda_mem_cache_lru_fifo_init;
  proc->cache.destroy = &kaapi_cuda_mem_cache_lru_fifo_destroy;
  proc->cache.insert = &kaapi_cuda_mem_cache_lru_fifo_insert;
  proc->cache.remove = &kaapi_cuda_mem_cache_lru_fifo_remove;
  proc->cache.is_full = &kaapi_cuda_mem_cache_lru_fifo_is_full;
  proc->cache.inc_use = &kaapi_cuda_mem_cache_lru_fifo_inc_use;
  proc->cache.dec_use = &kaapi_cuda_mem_cache_lru_fifo_dec_use;
}

static inline void kaapi_cuda_mem_cache_lru_double_fifo_config(kaapi_cuda_proc_t* proc)
{
  proc->cache.init = &kaapi_cuda_mem_cache_lru_double_fifo_init;
  proc->cache.destroy = &kaapi_cuda_mem_cache_lru_double_fifo_destroy;
  proc->cache.insert = &kaapi_cuda_mem_cache_lru_double_fifo_insert;
  proc->cache.remove = &kaapi_cuda_mem_cache_lru_double_fifo_remove;
  proc->cache.is_full = &kaapi_cuda_mem_cache_lru_double_fifo_is_full;
  proc->cache.inc_use = &kaapi_cuda_mem_cache_lru_double_fifo_inc_use;
  proc->cache.dec_use = &kaapi_cuda_mem_cache_lru_double_fifo_dec_use;
}

int kaapi_cuda_mem_cache_init(kaapi_cuda_proc_t* proc)
{
  const char* gpucache = getenv("KAAPI_GPU_CACHE_POLICY");
  if( gpucache != 0 )
  {
    if (strcmp(gpucache, "lru") == 0)
      kaapi_cuda_mem_cache_lru_fifo_config(proc);
    else if (strcmp(gpucache, "lru_double") == 0)
      kaapi_cuda_mem_cache_lru_double_fifo_config(proc);
    else {
      fprintf(stdout, "%s:%d:%s: *** Kaapi: bad value for 'KAAPI_GPU_CACHE_POLICY': '%s'\n",
              __FILE__, __LINE__, __FUNCTION__, gpucache );
      fflush(stdout);
      abort();
      return EINVAL;
    }
  }
  else
    kaapi_cuda_mem_cache_lru_double_fifo_config(proc);
  
  proc->cache.init( &proc->cache.data );
  
  return 0;
}

void kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t* proc)
{
  proc->cache.destroy( proc->cache.data );
}
