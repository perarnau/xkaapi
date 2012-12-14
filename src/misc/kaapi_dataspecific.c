/*
** xkaapi
** 
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_impl.h"
#endif

void* _kaapi_gettemporary_data(kaapi_processor_t*kproc, unsigned int id, size_t size)
{
  if (id >16) 
  {
    kaapi_assert_debug_m(1, "[kaapi] _kaapi_gettemporary_data, internal error: too many scratch arguments");
    return 0;
  }
  
  /* TODO: change all calls below to kaapi_memory_alloc/kaapi_memory_dealloc */
  if (kproc->size_specific[id] < size)
  {
#if defined(KAAPI_USE_CUDA)
    if( kaapi_processor_get_type(kaapi_get_current_processor()) == KAAPI_PROC_TYPE_CUDA )
    {
      if (kproc->data_specific[id] != 0)
          kaapi_cuda_mem_free(kaapi_make_localpointer(kproc->data_specific[id]));
      
      kproc->data_specific[id] = (void*)kaapi_cuda_mem_alloc(kaapi_memory_map_get_current_asid(), size,
                                                 (int)KAAPI_ACCESS_MODE_R);
    }
    else
#endif
    {
      if (kproc->data_specific[id] != 0)
        free(kproc->data_specific[id]);
  #if defined(KAAPI_DEBUG)
      kproc->data_specific[id] = calloc(1, size);
  #else
      kproc->data_specific[id] = malloc(size);
  #endif
    }
    kproc->size_specific[id] = size;
  }
  return kproc->data_specific[id];
}

void* kaapi_gettemporary_data(unsigned int id, size_t size)
{
  return _kaapi_gettemporary_data( kaapi_get_current_processor(), id, size);
}
