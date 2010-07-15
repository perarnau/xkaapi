/*
** kaapi_mem.h
** xkaapi
** 
** Created on Jul 2010
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


/* exported */

int kaapi_mem_map_initialize
(kaapi_mem_map_t* map, kaapi_mem_asid_t asid)
{
  map->asid = asid;
  map->head = NULL;
  return 0;
}

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

int kaapi_mem_map_find_or_insert
(kaapi_mem_map_t* map, kaapi_mem_addr_t addr, kaapi_mem_mapping_t** mapping)
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

int kaapi_mem_map_find_inverse
(kaapi_mem_map_t* map, kaapi_mem_addr_t raddr, kaapi_mem_mapping_t** mapping)
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

#if defined(KAAPI_USE_CUDA)
#if KAAPI_USE_CUDA

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

static inline int get_cuda_context_by_asid
(kaapi_mem_asid_t asid, CUcontext* ctx)
{
  /* ok since asid == kid */
  *ctx = kaapi_all_kprocessors[asid]->cuda_proc.ctx;
  return 0;
}

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

static int kaapi_mem_map_find_with_asid
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

  memcpy_dtoh(self_proc, (void*)hostptr, devptr, size);
}

void kaapi_mem_synchronize2(kaapi_mem_addr_t hostptr, size_t size)
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

  /* already valid on the host */
  if (!kaapi_mem_mapping_is_dirty(mapping, self_asid))
    return ;

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

  /* host addr no longer dirty */
  kaapi_mem_mapping_clear_dirty(mapping, self_asid);
}

#endif
#endif /* KAAPI_USE_CUDA */
