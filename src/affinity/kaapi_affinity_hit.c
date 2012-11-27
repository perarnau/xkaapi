/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** joao.lima@imag.fr
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

#include "kaapi_affinity.h"

#include "kaapi_tasklist.h"


/* return the sum of data valid in the kproc asid */
uint64_t kaapi_data_get_affinity_hit_size(
       const kaapi_processor_t * kproc,
       kaapi_taskdescr_t * td
)
{
  /* TODO */
#if 0
  int i;
  kaapi_mem_data_t *kmd;
  uint64_t hit = 0;
  void *sp = td->task->sp;

  if (td->fmt == NULL)
    return 0;
  
  const kaapi_mem_host_map_t* kproc_map = &kproc->mem_host_map;
  kaapi_mem_asid_t kproc_asid = kaapi_mem_host_map_get_asid(kproc_map);
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);

  
  for (i = 0; i < count_params; i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(td->fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    if (m == KAAPI_ACCESS_MODE_W)
    {
      kaapi_memory_view_t view = kaapi_format_get_view_param(td->fmt, i, sp);
      hit += kaapi_memory_view_size(&view);
    }
    else 
    {
      kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
      kaapi_data_t* data = kaapi_data(kaapi_data_t, &access);
      kmd = data->kmd;
      kaapi_assert_debug(kmd != 0);
      if (kaapi_bitmap_get_64(&kmd->valid_bits, kproc_asid))
      {
        kaapi_memory_view_t view = kaapi_format_get_view_param(td->fmt, i, sp);
        hit += kaapi_memory_view_size(&view);
      } 
    }
  }
  
  return hit;
#endif
  return 0;
}
