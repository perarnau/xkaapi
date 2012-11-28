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
  kaapi_metadata_info_set_dirty(kmdi, kasid);

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
  kaapi_memory_increase_access_view(kasid, &kdest->ptr, &kdest->view, (int)m);
  
  return kdest;
}

/**
  Returns 0 if no data was transfered, 1 otherwise.
 */
static inline int kaapi_memory_taskdescr_prologue_transfer(
                                                            kaapi_metadata_info_t* const kmdi,
                                                            kaapi_pointer_t dest,
                                                            kaapi_memory_view_t* const view_dest,
                                                            kaapi_pointer_t const src,
                                                            kaapi_memory_view_t* const view_src
                                                            )
{
  if(kaapi_metadata_info_is_valid(kmdi, kaapi_pointer2asid(dest)))
    return 0;
  
  if( (kaapi_pointer2asid(dest) == kaapi_pointer2asid(src))         ||
      (!kaapi_metadata_info_is_valid(kmdi, kaapi_pointer2asid(src)))
     )
  {
    uint16_t lid = kaapi_metadata_info_first_valid(kmdi);
    kaapi_data_t* kdata = kaapi_metadata_info_get_data_by_lid(kmdi, lid);
    kaapi_memory_copy(dest, view_dest, kdata->ptr, &kdata->view);
#if 0
    fprintf(stdout, "[%s] kid=%d kmdi=@%p dest=@%p(%d) src=@%p(%d) valid=@%p(%d) size=%lu\n",
            __FUNCTION__, kaapi_get_self_kid(), (void*)kmdi,
            kaapi_pointer2void(dest), kaapi_memory_address_space_getlid(kaapi_pointer2asid(dest)),
            kaapi_pointer2void(src), kaapi_memory_address_space_getlid(kaapi_pointer2asid(src)),
            kaapi_pointer2void(kdata->ptr), lid,
            kaapi_memory_view_size(view_src)
          );
    fflush(stdout);
#endif
  }
  else
  {
      kaapi_memory_copy(dest, view_dest, src, view_src);
#if 0
    fprintf(stdout, "[%s] kid=%d kmdi=@%p dest=@%p(%d) src=@%p(%d) size=%lu\n",
            __FUNCTION__, kaapi_get_self_kid(), (void*)kmdi,
            kaapi_pointer2void(dest), kaapi_memory_address_space_getlid(kaapi_pointer2asid(dest)),
            kaapi_pointer2void(src), kaapi_memory_address_space_getlid(kaapi_pointer2asid(src)),
            kaapi_memory_view_size(view_src)
            );
    fflush(stdout);
#endif
    
  }
  kaapi_metadata_info_clear_dirty(kmdi, kaapi_pointer2asid(dest));
  
  return 0;
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

  if (KAAPI_ACCESS_IS_READ(m))
  {
    kaapi_memory_taskdescr_prologue_transfer(kmdi, kdest->ptr, &kdest->view, ksrc->ptr, &ksrc->view);
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

static inline void kaapi_memory_taskdescr_epilogue_validate(
                                                            kaapi_taskdescr_t* td,
                                                            const int i,
                                                            const kaapi_access_mode_t m
                                                            )
{
  void* sp = td->task->sp;
  kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
  kaapi_data_t* ksrc = kaapi_data(kaapi_data_t, &access);
  kaapi_address_space_id_t kasid = kaapi_memory_map_get_current_asid();
  
  kaapi_memory_decrease_access_view(kasid, &ksrc->ptr, &ksrc->view, (int)m);
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

    kaapi_memory_taskdescr_epilogue_validate(td, i, m);
  }
  
  return 0;
}

int kaapi_memory_taskdescr_has_valid_writer(const kaapi_processor_t* kproc, kaapi_taskdescr_t* const td)
{
  size_t i;
  void* sp = td->task->sp;
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);

  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m = kaapi_format_get_mode_param(td->fmt, i, sp);
    
    m = KAAPI_ACCESS_GET_MODE(m);
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
      kaapi_data_t* kdata = kaapi_data(kaapi_data_t, &access);
      kaapi_metadata_info_t* kmdi = kdata->mdi;
      kaapi_assert_debug(kmdi != 0);
      kaapi_address_space_id_t kasid = kaapi_memory_map_kid2asid(kproc->kid);
      if(kaapi_metadata_info_is_valid(kmdi, kasid))
      {
        return 1;
      }
    }
  }
  
  return 0;
}

kaapi_processor_t *kaapi_memory_taskdescr_affinity_find_valid_wr(
                                                 kaapi_processor_t * kproc,
                                                 kaapi_taskdescr_t * td
                                                 )
{
  size_t i;
  void* sp = td->task->sp;
  const size_t count_params = kaapi_format_get_count_params(td->fmt, sp);
  kaapi_address_space_id_t kasid = kaapi_memory_map_kid2asid(kproc->kid);
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m = kaapi_format_get_mode_param(td->fmt, i, sp);
    
    m = KAAPI_ACCESS_GET_MODE(m);
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      kaapi_access_t access = kaapi_format_get_access_param(td->fmt, i, sp);
      kaapi_data_t* kdata = kaapi_data(kaapi_data_t, &access);
      kaapi_metadata_info_t* kmdi = kdata->mdi;
      kaapi_assert_debug(kmdi != 0);
      if(!kaapi_metadata_info_is_valid(kmdi, kasid))
      {
        uint16_t valid_lid = kaapi_metadata_info_first_valid(kmdi);
        if((valid_lid != 0) && (valid_lid < KAAPI_MAX_ADDRESS_SPACE)){
          /* TODO: check if valid_lid(target) can execute this task. */
#if 0
          fprintf(stdout, "[%s] (kid=%d) kid=%lu td=%p(name=%s) "
                  "src_asid=%lu (kid=%lu) to dest_asid=%lu (kid=%lu)\n",
                  __FUNCTION__,
                  kaapi_get_self_kid(),
                  (long unsigned int) kproc->kid,
                  (void *) td, td->fmt->name,
                  (long unsigned int) kaapi_memory_address_space_getlid(kasid),
                  (long unsigned int) kaapi_memory_map_asid2kid(kasid),
                  (long unsigned int) valid_lid,
                  (long unsigned int) kaapi_memory_map_lid2kid(valid_lid)
                  );
          fflush(stdout);
#endif
          return kaapi_all_kprocessors[kaapi_memory_map_lid2kid(valid_lid)];
        }
      }
    }
  }
  
  return kproc;
}
