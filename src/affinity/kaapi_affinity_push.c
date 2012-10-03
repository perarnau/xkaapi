/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
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


kaapi_processor_t *kaapi_push_by_affinity_default(
  kaapi_processor_t * kproc,
	kaapi_taskdescr_t * td
)
{
  return kproc;
}

kaapi_processor_t *kaapi_push_by_affinity_locality(
                                             kaapi_processor_t * kproc,
                                             kaapi_taskdescr_t * td
)
{
  int i;
  kaapi_mem_data_t *kmd;
  void *sp;
  sp = td->task->sp;
  if (td->fmt == NULL)
    return kproc;
  
  //  kaapi_mem_host_map_t* host_map = kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  //  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_host_map_t *local_map = kaapi_get_current_mem_host_map();
  kaapi_mem_asid_t local_asid = kaapi_mem_host_map_get_asid(local_map);
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);
  size_t devices[KAAPI_MEM_ASID_MAX];
  kaapi_bitmap_value64_t dev_bitmap;
  int dev;
  int current_dev = 0;
  size_t current_dev_size = 0;
  
  memset(devices, 0, KAAPI_MEM_ASID_MAX * sizeof(size_t));
  for (i = 0; i < count_params; i++) 
  {
    kaapi_access_mode_t m =
    KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(td->fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
    kaapi_data_t *data = kaapi_data(kaapi_data_t, &access);
    kmd = data->kmd;
    kaapi_assert_debug(kmd != 0);
    kaapi_bitmap_copy_64(&dev_bitmap, &kmd->valid_bits);
    while ((dev = kaapi_bitmap_value_first1_and_zero_64(&dev_bitmap)) != 0) 
    {
      dev--;
      kaapi_data_t *const dev_data =
      (kaapi_data_t *) kaapi_mem_data_get_addr(kmd, dev);
      devices[dev] += kaapi_memory_view_size(&dev_data->view);
      if (devices[dev] > current_dev_size) {
        current_dev = dev;
        current_dev_size = devices[dev];
      }
    }
  }
  
  if ((current_dev != 0) && (current_dev != local_asid)) 
  {
#if 0
    fprintf(stdout, "[%s] kid=%lu td=%p(name=%s) "
            "src_asid=%lu (kid=%lu) to dest_asid=%lu (kid=%lu) size=%lu\n",
            __FUNCTION__,
            (long unsigned int) kproc->kid,
            (void *) td, td->fmt->name,
            (long unsigned int) local_asid,
            (long unsigned int) kaapi_mem_asid2kid(local_asid),
            (long unsigned int) current_dev,
            (long unsigned int) kaapi_mem_asid2kid(current_dev),
            current_dev_size);
    fflush(stdout);
#endif
    return kaapi_all_kprocessors[kaapi_mem_asid2kid(current_dev)];
  }
  return kproc;
}

kaapi_processor_t *kaapi_push_by_affinity_rand(kaapi_processor_t * kproc,
                                       kaapi_taskdescr_t * td)
{
  int nbproc, procid;
  
  if( td->fmt == 0 )
    return kproc;
  
  if (kproc->fnc_selecarg[0] == 0) 
    kproc->fnc_selecarg[0] = (uintptr_t)(long)rand();
  
  nbproc = kaapi_count_kprocessors;
redo_select:
  procid = rand_r( (unsigned int*)&kproc->fnc_selecarg ) % nbproc;
  if( procid == 0 )
    goto redo_select;
  
  return kaapi_all_kprocessors[ procid ];
}

kaapi_processor_t *kaapi_push_by_affinity_writer(kaapi_processor_t * kproc,
					      kaapi_taskdescr_t * td)
{
  int i;
  kaapi_mem_data_t *kmd;
  void *sp;
  sp = td->task->sp;
  if (td->fmt == NULL)
    return kproc;

  kaapi_mem_host_map_t *local_map = kaapi_get_current_mem_host_map();
  kaapi_mem_asid_t local_asid = kaapi_mem_host_map_get_asid(local_map);
  kaapi_mem_asid_t valid_asid;
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);

  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m =
	KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(td->fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;

    if (KAAPI_ACCESS_IS_WRITE(m)) {
      kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
      kaapi_data_t *data = kaapi_data(kaapi_data_t, &access);
      kmd = data->kmd;
      kaapi_assert_debug(kmd != 0);
      if (kaapi_mem_data_is_dirty(kmd, local_asid)) {
	valid_asid = kaapi_mem_data_get_nondirty_asid(kmd);
	if ((valid_asid != 0) && (valid_asid != local_asid)) {
#if 0
	  fprintf(stdout, "[%s] kid=%lu td=%p(name=%s) "
		  "src_asid=%lu (kid=%lu) to dest_asid=%lu (kid=%lu)\n",
		  __FUNCTION__,
		  (long unsigned int) kproc->kid,
		  (void *) td, td->fmt->name,
		  (long unsigned int) local_asid,
		  (long unsigned int) kaapi_mem_asid2kid(local_asid),
		  (long unsigned int) valid_asid,
		  (long unsigned int) kaapi_mem_asid2kid(valid_asid),
		  valid_asid);
	  fflush(stdout);
#endif
	  return kaapi_all_kprocessors[kaapi_mem_asid2kid(valid_asid)];
	}
      }
    }
  }

  return kproc;
}

int kaapi_data_get_affinity_is_valid_writer(
     const kaapi_processor_t * kproc, kaapi_taskdescr_t * td)
{
  int i;
  kaapi_mem_data_t *kmd;
  void *sp;
  sp = td->task->sp;

  kaapi_mem_host_map_t *local_map = kaapi_get_current_mem_host_map();
  kaapi_mem_asid_t local_asid = kaapi_mem_host_map_get_asid(local_map);
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);

  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m =
	KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(td->fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;

    if (KAAPI_ACCESS_IS_WRITE(m)) {
      kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
      kaapi_data_t *data = kaapi_data(kaapi_data_t, &access);
      kmd = data->kmd;
      kaapi_assert_debug(kmd != 0);
      if (!kaapi_mem_data_is_dirty(kmd, local_asid)) {
#if 0
	fprintf(stdout, "[%s]: asid=%d valid (td=%p,name=%s)\n",
	    __FUNCTION__,
	    local_asid,
	    (void*)td,
	    td->fmt->name );
	fflush(stdout);
#endif
	return 1;
      }
    }
  }

  return 0;
}
