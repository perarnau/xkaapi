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

static inline kaapi_data_t* kaapi_memory_taskdescr_prologue_alloc(
                                                         kaapi_metadata_info_t* kmdi,
                                                         kaapi_address_space_id_t kasid,
                                                         kaapi_data_t* kdata,
                                                         const kaapi_access_mode_t m
                                                         )
{
  kaapi_memory_view_t view = kdata->view;
  kaapi_pointer_t ptr = kaapi_memory_allocate_view(kasid, &view, (int)m);
  kaapi_memory_bind_view_with_metadata(kasid, kmdi, 0, kaapi_pointer2void(ptr), &view);

  return kaapi_metadata_info_get_data(kmdi, kasid);
}

static inline kaapi_data_t* kaapi_memory_taskdescr_prologue_access(
                                                                  kaapi_metadata_info_t* kmdi,
                                                                  kaapi_address_space_id_t kasid,
                                                                  kaapi_data_t* kdata,
                                                                  const kaapi_access_mode_t m
                                                                  )
{
  kaapi_data_t* kdest = kaapi_metadata_info_get_data(kmdi, kasid);
  kaapi_memory_access_view(kasid, &kdest->ptr, &kdest->view, (int)m);
  
  return kdest;
}

static inline void kaapi_memory_taskdescr_prologue_validate(
                                                             kaapi_taskdescr_t* td,
                                                             const int i,
                                                             const kaapi_access_mode_t m
                                                             )
{
  void* sp = td->task->sp;
  kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
  kaapi_data_t* ksrc = kaapi_data(kaapi_data_t, &access);
  kaapi_data_t* kdest;
  kaapi_metadata_info_t* kmdi = ksrc->mdi;
  kaapi_address_space_id_t kasid = kaapi_memory_map_get_current_asid();
  
  if(!kaapi_metadata_info_has_data(kmdi, kasid))
  {
    kdest = kaapi_memory_taskdescr_prologue_alloc(kmdi, kasid, ksrc, m);
  }
  else
  {
    /* increment pointer usage */
    kdest = kaapi_memory_taskdescr_prologue_access(kmdi, kasid, ksrc, m);
  }
  
  if (KAAPI_ACCESS_IS_WRITE(m))
  {
    kaapi_metadata_info_set_all_dirty_except(kmdi, kasid);
  }
  
  if(kaapi_memory_address_space_gettype(kasid) != KAAPI_MEM_TYPE_CPU)
  {
    /* sets new pointer to the task */
    access.data = kdest;
    kaapi_format_set_access_param(td->fmt, i, sp, &access);
  }
}

int kaapi_memory_taskdescr_prologue(kaapi_taskdescr_t * td)
{
  size_t i;
  void *sp;
  
  if ( td->fmt == 0 )
    return 0;
  
  sp = td->task->sp;
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);
  
  
#if 0
  fprintf(stdout, "[%s] params=%ld kid=%lu asid=%d\n", __FUNCTION__,
          count_params,
          (unsigned long) kaapi_get_current_kid(),
          kaapi_memory_address_space_getlid(kaapi_memory_map_get_current_asid()));
  fflush(stdout);
#endif
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m = kaapi_format_get_mode_param(td->fmt, i, sp);
    m = KAAPI_ACCESS_GET_MODE(m);
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
     kaapi_memory_taskdescr_prologue_validate(td, i, m);
  }
  
  return 0;
}

int kaapi_memory_taskdescr_epilogue(kaapi_taskdescr_t * td)
{
  size_t i;
  void *sp;
  
  if ( td->fmt == 0 )
    return 0;
  
  sp = td->task->sp;
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);
#if 0
  fprintf(stdout, "[%s] params=%ld kid=%lu asid=%d\n", __FUNCTION__,
          count_params,
          (unsigned long) kaapi_get_current_kid(),
          kaapi_memory_address_space_getlid(kaapi_memory_map_get_current_asid()));
  fflush(stdout);
#endif
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m = kaapi_format_get_mode_param(td->fmt, i, sp);
    
    m = KAAPI_ACCESS_GET_MODE(m);
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
    kaapi_data_t *kdata = kaapi_data(kaapi_data_t, &access);
    
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      //
    }
  }
  
  return 0;
}
