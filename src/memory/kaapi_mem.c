/*
** kaapi_mem.h
** xkaapi
** 
**
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include "kaapi_mem.h"

#if defined(KAAPI_USE_CUDA)

#include <cuda.h>
#include "../machine/cuda/kaapi_cuda_error.h"

/* todo: portability layer should handle this */
static inline int memcpy_dtoh
(kaapi_processor_t* proc, void* hostptr, CUdeviceptr devptr, size_t size)
{
#if 0 /* async version */
  const CUresult res = cuMemcpyDtoHAsync
    (hostptr, devptr, size, proc->cuda_proc.stream);
#else
  const CUresult res = cuMemcpyDtoH
    (hostptr, devptr, size);
#endif

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyDToHAsync", res);
    return -1;
  }

  return 0;
}

extern kaapi_processor_t* get_proc_by_asid(kaapi_mem_asid_t);

static inline int get_cuda_context_by_asid
(kaapi_mem_asid_t asid, CUcontext* ctx)
{
  /* todo, lock */

  kaapi_processor_t* const kproc = get_proc_by_asid(asid);
  kaapi_assert_debug(kproc);

  *ctx = kproc->cuda_proc.ctx;
  return 0;
}

static inline void put_cuda_context_by_asid
(kaapi_mem_asid_t asid, CUcontext ctx)
{
  /* todo, lock */
}

#if 0 //* does not compile with default configure
static inline kaapi_mem_map_t* get_proc_mem_map(kaapi_processor_t* proc)
{
  return &proc->mem_map;
}

static inline kaapi_mem_map_t* get_host_mem_map(void)
{
  return get_proc_mem_map(kaapi_all_kprocessors[0]);
}

static inline kaapi_mem_map_t* get_self_mem_map(void)
{
  return get_proc_mem_map(kaapi_get_current_processor());
}
#endif

int kaapi_mem_map_find_with_asid
(kaapi_mem_map_t* map, kaapi_mem_addr_t addr,
 kaapi_mem_asid_t asid, kaapi_mem_mapping_t** mapping)
{
  /* find a mapping in the map such that addrs[asid] == addr */

  kaapi_mem_mapping_t* pos;

  *mapping = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    if (!kaapi_mem_mapping_has_addr(pos, asid))
      continue ;

    if (kaapi_mem_mapping_get_addr(pos, asid) == addr)
    {
      *mapping = pos;
      return 0;
    }
  }

  return -1;
}

static kaapi_cuda_proc_t* get_cu_context(void)
{
  size_t count = kaapi_count_kprocessors;
  kaapi_processor_t** proc = kaapi_all_kprocessors;
  CUresult res;

  for (; count; --count, ++proc)
  {
    if ((*proc)->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      kaapi_cuda_proc_t* const cu_proc = &(*proc)->cuda_proc;

      pthread_mutex_lock(&cu_proc->ctx_lock);
      res = cuCtxPushCurrent(cu_proc->ctx);
      if (res == CUDA_SUCCESS) return cu_proc;
      pthread_mutex_unlock(&cu_proc->ctx_lock);
    }
  }

  return NULL;
}

static void put_cu_context(kaapi_cuda_proc_t* cu_proc)
{
  cuCtxPopCurrent(&cu_proc->ctx);
  pthread_mutex_unlock(&cu_proc->ctx_lock);
}

void* kaapi_mem_alloc_host(size_t size)
{
  /* allocate host memory needed for async transfers.
     the first cuda mem asid is used since a context
     is needed to perform the allocation.
     todo: create a dedicated one for the host
   */

  kaapi_cuda_proc_t* const cu_proc = get_cu_context();

  void* hostptr;
  CUresult res;

  if (cu_proc == NULL) return NULL;

  res = cuMemHostAlloc(&hostptr, size, CU_MEMHOSTALLOC_PORTABLE);

  put_cu_context(cu_proc);

  if (res != CUDA_SUCCESS) return NULL;

  return hostptr;
}

void kaapi_mem_free_host(void* hostptr)
{
  kaapi_cuda_proc_t* const cu_proc = get_cu_context();
  if (cu_proc == NULL) return ;
  cuMemFreeHost(hostptr);
  put_cu_context(cu_proc);
}

#endif /* KAAPI_USE_CUDA */


/* exported */

void kaapi_mem_map_cleanup(kaapi_mem_map_t* map)
{
  kaapi_mem_mapping_t* pos = map->head;

  while (pos != NULL)
  {
    kaapi_mem_mapping_t* const tmp = pos;
    pos = pos->next;
    free(tmp);
  }
  map->head = NULL;
}


int kaapi_mem_map_find
(kaapi_mem_map_t* map, kaapi_mem_addr_t addr, kaapi_mem_mapping_t** mapping)
{
  /* find a mapping in the map such that addrs[map->asid] == addr. */

  kaapi_mem_mapping_t* pos;

  *mapping = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    /* assume pos->addrs[map->asid] always set */
    if (kaapi_mem_mapping_get_addr(pos, map->asid) == addr)
    {
      *mapping = pos;
      return 0;
    }
  }

  return -1;
}

int kaapi_mem_map_find_or_insert(kaapi_mem_map_t* map, kaapi_mem_addr_t addr, kaapi_mem_mapping_t** mapping)
{
  /* see comments in the above function. if no mapping is found, create one. */
  const int res = kaapi_mem_map_find(map, addr, mapping);
  if (res != -1)
    return 0;

  *mapping = malloc(sizeof(kaapi_mem_mapping_t));
  if (*mapping == NULL)
    return -1;

  /* identity mapping */
  kaapi_mem_mapping_init_identity(*mapping, map->asid, addr);

  /* link against others */
  (*mapping)->next = map->head;
  map->head = *mapping;

  return 0;
}

int kaapi_mem_map_find_inverse(
  kaapi_mem_map_t*      map, 
  kaapi_mem_addr_t      raddr __attribute__((unused)), 
  kaapi_mem_mapping_t** mapping
)
{
  /* given a remote address, find the
     corresponding host address. greedy
     inverted search, exhaust all the
     memory mapping set.
   */
  kaapi_mem_mapping_t* pos;
  kaapi_mem_asid_t asid;

  *mapping = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
    {
      if (!kaapi_mem_mapping_has_addr(pos, asid))
	continue ;

      *mapping = pos;
      return 0;
    }
  }

  return -1;
}

kaapi_mem_asid_t kaapi_mem_mapping_get_nondirty_asid
(const kaapi_mem_mapping_t* mapping)
{
  /* assuming there is one, find an asid where addr is valid. */

  kaapi_mem_asid_t asid;

  for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
    if (!kaapi_mem_mapping_is_dirty(mapping, asid))
      break ;

  return asid;
}

void kaapi_mem_delete_host_mappings
(kaapi_mem_addr_t addr, size_t size)
{
  /* delete all the host mappings on [addr, size[
     this function deallocates any remote memory
     but it is up to the host to deallocate the
     local memory if needed.
  */

  /* non inclusive */
  const kaapi_mem_addr_t last_addr = addr + size;

  kaapi_processor_t* const kproc = kaapi_all_kprocessors[0];

  kaapi_mem_mapping_t* pos = kproc->mem_map.head;
  kaapi_mem_mapping_t* prev = NULL;

  while (pos != NULL)
  {
    kaapi_mem_asid_t asid;

    kaapi_mem_mapping_t* const tmp = pos;
    pos = pos->next;

    /* skip if does not map the host addr */

    if (!((tmp->addrs[0] >= addr) && (tmp->addrs[0] < last_addr)))
    {
      prev = tmp;
      continue ;
    }

    /* free remote mem and delete the mapping */

    for (asid = 1; asid < KAAPI_MEM_ASID_MAX; ++asid)
    {
      if (kaapi_mem_mapping_has_addr(tmp, asid) == 0)
	continue ;

#if defined(KAAPI_DEBUG)
      printf("delete: %u::0x%lx\n", asid, (unsigned long)tmp->addrs[asid]);
#endif

      /* assume the remote device is a gpu.
	 todo: asid_to_kproc[] map
      */
#if defined(KAAPI_USE_CUDA)
      {
	CUresult res;
	CUcontext ctx;

	get_cuda_context_by_asid(asid, &ctx);

	res = cuCtxPushCurrent(ctx);
	if (res == CUDA_SUCCESS)
	{
	  res = cuMemFree(tmp->addrs[asid]);
#if defined(KAAPI_DEBUG)
	  if (res != CUDA_SUCCESS)
	    kaapi_cuda_error("cuMemFree", res);
#endif
	  cuCtxPopCurrent(&ctx);
	}
#if defined(KAAPI_DEBUG)
	else
	  kaapi_cuda_error("cuCtxPushCurrent", res);
#endif

	put_cuda_context_by_asid(asid, ctx);
      }
#endif /* KAAPI_USE_CUDA */
    }

    /* unlink tmp the node */
    if (prev == NULL)
      kproc->mem_map.head = tmp->next;
    else
      prev->next = tmp->next;
    free(tmp);
  }
}

#if defined(KAAPI_USE_CUDA)

void kaapi_mem_synchronize(kaapi_mem_addr_t devptr, size_t size)
{
  /* ensure everything past this point
     has been written to host memory.
     assumed to be called from gpu thread.
   */

  kaapi_processor_t* const self_proc = kaapi_get_current_processor();

  kaapi_mem_map_t* const host_map = get_host_mem_map();
  kaapi_mem_map_t* const self_map = get_proc_mem_map(self_proc);

  const kaapi_mem_asid_t host_asid = host_map->asid;
  const kaapi_mem_asid_t self_asid = self_map->asid;

  kaapi_mem_addr_t hostptr;
  kaapi_mem_mapping_t* mapping;

  /* find hostptr, devptr mapping. assume no error. */
  kaapi_mem_map_find_with_asid(host_map, devptr, self_asid, &mapping);
  hostptr = kaapi_mem_mapping_get_addr(mapping, host_asid);

#if defined(KAAPI_DEBUG)
  printf("memcpy_dtoh(%lx, %lx, %u)\n",
	 (uintptr_t)hostptr, (uintptr_t)devptr, size);
#endif

  memcpy_dtoh(self_proc, (void*)hostptr, devptr, size);
}

int kaapi_mem_synchronize2(kaapi_mem_addr_t hostptr, size_t size)
{
  /* same as above but to be called on host side */

  kaapi_processor_t* const self_proc = kaapi_get_current_processor();
  kaapi_mem_map_t* const self_map = get_proc_mem_map(self_proc);
  const kaapi_mem_asid_t self_asid = self_map->asid;

  kaapi_mem_mapping_t* mapping;
  kaapi_mem_asid_t asid;
  kaapi_mem_addr_t devptr;

  CUresult res;
  CUcontext dev_ctx;

  /* find a valid space associated with device */
  kaapi_mem_map_find(self_map, hostptr, &mapping);
  if (mapping == NULL)
    return -1;

  /* already valid on the host */
  if (!kaapi_mem_mapping_is_dirty(mapping, self_asid))
    return 0;

  /* get valid remote pointer */
  asid = kaapi_mem_mapping_get_nondirty_asid(mapping);
  devptr = kaapi_mem_mapping_get_addr(mapping, asid);

  /* set the device context */
  get_cuda_context_by_asid(asid, &dev_ctx);

  res = cuCtxPushCurrent(dev_ctx);
  if (res == CUDA_SUCCESS)
  {
    memcpy_dtoh(self_proc, (void*)hostptr, (CUdeviceptr)devptr, size);
    cuCtxPopCurrent(&dev_ctx);
  }
  else
  {
    kaapi_cuda_error("cuCtxPushCurrent", res);
  }

  put_cuda_context_by_asid(asid, dev_ctx);

  /* host addr no longer dirty */
  kaapi_mem_mapping_clear_dirty(mapping, self_asid);

  return 0;
}

int kaapi_mem_synchronize3(kaapi_mem_mapping_t* mapping, size_t size)
{
  /* called by the host to validate mapping */

  kaapi_processor_t* const self_proc = kaapi_get_current_processor();
  kaapi_mem_map_t* const self_map = get_proc_mem_map(self_proc);
  const kaapi_mem_asid_t self_asid = self_map->asid;

  kaapi_mem_addr_t hostptr = mapping->addrs[self_asid];

  kaapi_mem_asid_t asid;
  kaapi_mem_addr_t devptr;

  CUresult res;
  CUcontext dev_ctx;

  /* get valid remote pointer */
  asid = kaapi_mem_mapping_get_nondirty_asid(mapping);
  devptr = kaapi_mem_mapping_get_addr(mapping, asid);

  /* set the device context */
  get_cuda_context_by_asid(asid, &dev_ctx);

  res = cuCtxPushCurrent(dev_ctx);
  if (res == CUDA_SUCCESS)
  {
    memcpy_dtoh(self_proc, (void*)hostptr, (CUdeviceptr)devptr, size);
    cuCtxPopCurrent(&dev_ctx);
  }
  else
  {
    kaapi_cuda_error("cuCtxPushCurrent", res);
  }

  put_cuda_context_by_asid(asid, dev_ctx);

  /* host addr no longer dirty */
  kaapi_mem_mapping_clear_dirty(mapping, self_asid);

  return 0;
}

#endif /* KAAPI_USE_CUDA */


#if 0 /* TEMP_FUNCTIONS */

void* kaapi_mem_find_host_addr(kaapi_mem_addr_t addr)
{
  /* addr the address a mapping is looked for in the host */

  /* self info */
  kaapi_processor_t* const self_proc = kaapi_get_current_processor();
  kaapi_mem_map_t* const self_map = get_proc_mem_map(self_proc);
  const kaapi_mem_asid_t self_asid = self_map->asid;

  kaapi_mem_map_t* const host_map = get_host_mem_map();

  kaapi_mem_mapping_t* pos;
  for (pos = host_map->head; pos != NULL; pos = pos->next)
  {
    if (kaapi_mem_mapping_has_addr(pos, self_asid))
    {
      if (kaapi_mem_mapping_get_addr(pos, self_asid) == addr)
	return (void*)kaapi_mem_mapping_get_addr(pos, host_map->asid);
    }
  }

  return NULL;
}

#endif /* TEMP_FUNCTIONS */
