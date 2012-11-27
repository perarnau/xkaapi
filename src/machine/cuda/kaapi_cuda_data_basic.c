
#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_memory.h"	/* TODO: remove this */
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_data_basic.h"

static inline void
kaapi_cuda_data_basic_view_convert(kaapi_memory_view_t * dest_view, const
                                   kaapi_memory_view_t * src_view)
{
  switch (src_view->type) {
    case KAAPI_MEMORY_VIEW_1D:
      dest_view->size[0] = src_view->size[0];
      break;
    case KAAPI_MEMORY_VIEW_2D:
      dest_view->size[0] = src_view->size[0];
      dest_view->size[1] = src_view->size[1];
      dest_view->lda = src_view->size[1];
      break;
    default:
      kaapi_assert(0);
      break;
  }
  dest_view->wordsize = src_view->wordsize;
  dest_view->type = src_view->type;
}

static inline kaapi_data_t
*xxx_kaapi_cuda_data_basic_allocate(kaapi_mem_host_map_t * cuda_map,
                                    kaapi_mem_data_t * kmd,
                                    kaapi_data_t * src)
{
  const kaapi_mem_asid_t asid = kaapi_mem_host_map_get_asid(cuda_map);
  kaapi_data_t *dest = (kaapi_data_t *) calloc(1, sizeof(kaapi_data_t));
  kaapi_mem_addr_t addr;
  addr = (kaapi_mem_addr_t)kaapi_cuda_mem_alloc_(kaapi_memory_view_size(&src->view));
  dest->ptr = kaapi_make_pointer(0, (void *) addr);
  kaapi_cuda_data_basic_view_convert(&dest->view, &src->view);
  kaapi_mem_data_set_addr(kmd, asid, (kaapi_mem_addr_t) dest);
  kaapi_mem_host_map_find_or_insert_(cuda_map, (kaapi_mem_addr_t)
                                     kaapi_pointer2void(dest->ptr), &kmd);
  return dest;
}

int kaapi_cuda_data_basic_allocate(kaapi_format_t * fmt, void *sp)
{
  const size_t count_params = kaapi_format_get_count_params(fmt, sp);
  size_t i;
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  kaapi_mem_host_map_t *host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_data_t *kmd;
  
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
          count_params,
          (unsigned long) kaapi_get_current_kid(),
          (unsigned int long) kaapi_mem_host_map_get_asid(cuda_map));
  fflush(stdout);
#endif
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m =
    KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    kaapi_access_t access = kaapi_format_get_access_param(fmt,
                                                          i, sp);
    kaapi_data_t *src = kaapi_data(kaapi_data_t, &access);
    kaapi_mem_host_map_find_or_insert(host_map, (kaapi_mem_addr_t)
                                      kaapi_pointer2void(src->ptr), &kmd);
    kaapi_assert_debug(kmd != 0);
    
    if (!kaapi_mem_data_has_addr(kmd, host_asid))
      kaapi_mem_data_set_addr(kmd, host_asid, (kaapi_mem_addr_t) src);
    
    kaapi_data_t *dest =
    xxx_kaapi_cuda_data_basic_allocate(cuda_map, kmd, src);
    
    /* sets new pointer to the task */
    access.data = dest;
    kaapi_format_set_access_param(fmt, i, sp, &access);
  }
  
  return 0;
}


int kaapi_cuda_data_basic_send(kaapi_format_t * fmt, void *sp)
{
  const size_t count_params = kaapi_format_get_count_params(fmt, sp);
  size_t i;
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  kaapi_mem_host_map_t *host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
          count_params,
          (unsigned long) kaapi_get_current_kid(),
          (unsigned int long) kaapi_mem_host_map_get_asid(cuda_map));
  fflush(stdout);
#endif
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m =
    KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    if (KAAPI_ACCESS_IS_READ(m)) {
      kaapi_mem_data_t *kmd;
      kaapi_access_t access = kaapi_format_get_access_param(fmt,
                                                            i, sp);
      kaapi_data_t *dev_data = kaapi_data(kaapi_data_t, &access);
      
      kaapi_mem_host_map_find_or_insert(cuda_map, (kaapi_mem_addr_t)
                                        kaapi_pointer2void(dev_data->ptr),
                                        &kmd);
      kaapi_assert_debug(kmd != 0);
      kaapi_data_t *host_data = (kaapi_data_t *)
      kaapi_mem_data_get_addr(kmd, host_asid);
      kaapi_cuda_mem_copy_htod(dev_data->ptr, &dev_data->view,
                               host_data->ptr, &host_data->view);
    }
  }
  
  return 0;
}

int kaapi_cuda_data_basic_recv(kaapi_format_t * fmt, void *sp)
{
  const size_t count_params = kaapi_format_get_count_params(fmt, sp);
  size_t i;
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
  kaapi_mem_host_map_t *host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_data_t *kmd;
  
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
          count_params,
          (unsigned long) kaapi_get_current_kid(),
          (unsigned int long) kaapi_mem_host_map_get_asid(cuda_map));
  fflush(stdout);
#endif
  
  for (i = 0; i < count_params; i++) {
    kaapi_access_mode_t m =
    KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, i, sp));
    if (m == KAAPI_ACCESS_MODE_V)
      continue;
    
    kaapi_access_t access = kaapi_format_get_access_param(fmt,
                                                          i, sp);
    kaapi_data_t *dev_data = kaapi_data(kaapi_data_t, &access);
    kaapi_mem_host_map_find_or_insert(cuda_map, (kaapi_mem_addr_t)
                                      kaapi_pointer2void(dev_data->ptr),
                                      &kmd);
    kaapi_assert_debug(kmd != 0);
    
    if (KAAPI_ACCESS_IS_WRITE(m)) {
      kaapi_data_t *host_data =
      (kaapi_data_t *) kaapi_mem_data_get_addr(kmd, host_asid);
      kaapi_cuda_mem_copy_dtoh(host_data->ptr, &host_data->view,
                               dev_data->ptr, &dev_data->view);
    }
    kaapi_mem_data_clear_addr(kmd, cuda_asid);
    kaapi_cuda_mem_free(dev_data->ptr);
    free(dev_data);
  }
  
  return 0;
}
