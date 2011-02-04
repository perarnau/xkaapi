/*
** xkaapi
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


/* memory write view switch */
typedef int (*write_func_t)(void*, void*, const void*, size_t);

static int kaapi_memory_write_view
(
 kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
 const void* src, const kaapi_memory_view_t* view_src,
 write_func_t write_fn, void* write_ctx
)
{
  int error = EINVAL;

  switch (view_src->type)
  {
    case KAAPI_MEMORY_VIEW_1D:
    {
      if (view_dest->type != KAAPI_MEMORY_VIEW_1D) goto on_error;
      if (view_dest->size[0] != view_src->size[0]) goto on_error;
      error = write_fn(write_ctx, (void*)dest, src, view_src->size[0]*view_src->wordsize );
    } break;

    case KAAPI_MEMORY_VIEW_2D:
    {
      const char* laddr;
      char* raddr;
      size_t size;

      if (view_dest->type != KAAPI_MEMORY_VIEW_2D) goto on_error;
      if (view_dest->size[0] != view_src->size[0]) goto on_error;

      size = view_src->size[0] * view_src->size[1];
      if (size != view_dest->size[0] * view_dest->size[1]) goto on_error;
      
      laddr = (const char*)src;
      raddr = (char*)dest;
      
      if (kaapi_memory_view_iscontiguous(view_src) &&
	  kaapi_memory_view_iscontiguous(view_dest))
      {
	error = write_fn(write_ctx, raddr, laddr, size * view_src->wordsize);
	if (error) goto on_error;
      }
      else 
      {
        size_t i;
        size_t size_row = view_src->size[1]*view_src->wordsize;
        size_t llda = view_src->lda * view_src->wordsize;
        size_t rlda = view_dest->lda * view_src->wordsize;

        kaapi_assert_debug( view_dest->size[1] == view_src->size[1] );

        for (i=0; i<view_src->size[0]; ++i, laddr += llda, raddr += rlda)
	{
	  error = write_fn(write_ctx, raddr, laddr, size_row);
	  if (error) goto on_error;
	}
      }
      break;
    }

    default: goto on_error; break;
  }

  /* success */
  error = 0;
 on_error:
  return error;
}

#if defined(KAAPI_USE_CUDA)

#include "../cuda/kaapi_cuda.h"
#include "../machine/cuda/kaapi_cuda_error.h"

static kaapi_cuda_proc_t* kasid_to_cuda_proc
(kaapi_address_space_id_t kasid)
{
  size_t count = kaapi_count_kprocessors;
  kaapi_processor_t** proc = kaapi_all_kprocessors;

  for (; count; --count, ++proc)
  {
    if ((*proc)->proc_type == KAAPI_PROC_TYPE_CUDA)
      return &(*proc)->cuda_proc;
  }

  return NULL;
}

#if 0
static inline unsigned int is_self_cu_proc
(kaapi_cuda_proc_t* cu_proc)
{
  if (cu_proc == &kaapi_get_current_processor()->cu_proc)
    return 1;
  return 0;
}
#endif

static kaapi_cuda_proc_t* get_cu_context
(kaapi_address_space_id_t kasid)
{
  kaapi_cuda_proc_t* const cu_proc = kasid_to_cuda_proc(kasid);
  CUresult res;

  if (cu_proc == NULL) return NULL;

#if 0 /* todo */
  /* self proc, dont acquire */
  if (is_self_cu_proc(cu_proc)) return cu_proc;
#endif

  pthread_mutex_lock(&cu_proc->ctx_lock);
  res = cuCtxPushCurrent(cu_proc->ctx);
  if (res == CUDA_SUCCESS) return cu_proc;
  pthread_mutex_unlock(&cu_proc->ctx_lock);
  return NULL;
}

static void put_cu_context(kaapi_cuda_proc_t* cu_proc)
{
  cuCtxPopCurrent(&cu_proc->ctx);
  pthread_mutex_unlock(&cu_proc->ctx_lock);
}

static inline int allocate_cu_mem(CUdeviceptr* devptr, size_t size)
{
  const CUresult res = cuMemAlloc(devptr, size);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemAlloc", res);
    return -1;
  }

  return 0;
}

static inline void free_cu_mem(CUdeviceptr devptr)
{
  cuMemFree(devptr);
}

static inline int memcpy_cu_htod
(kaapi_cuda_proc_t* cu_proc, CUdeviceptr devptr, const void* hostptr, size_t size)
{
#if 0 /* async version */
  const CUresult res = cuMemcpyHtoDAsync
    (devptr, hostptr, size, cu_proc->stream);
#else
  const CUresult res = cuMemcpyHtoD
    (devptr, hostptr, size);
#endif

#if 0
  printf("htod(%lx, %lx, %lx)\n", (uintptr_t)devptr, (uintptr_t)hostptr, size);
#endif

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyHToDAsync", res);
    return -1;
  }

  return 0;
}

static inline int memcpy_cu_dtoh
(kaapi_cuda_proc_t* cu_proc, void* hostptr, CUdeviceptr devptr, size_t size)
{
#if 0 /* async version */
  const CUresult res = cuMemcpyDtoHAsync
    (hostptr, devptr, size, cu_proc->stream);
#else
  const CUresult res = cuMemcpyDtoH
    (hostptr, devptr, size);
#endif

#if 0
  printf("dtoh(%lx, %lx, %lx)\n", (uintptr_t)hostptr, (uintptr_t)devptr, size);
#endif

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyDToHAsync", res);
    return -1;
  }

  return 0;
}

static inline int kaapi_memory_write_cu2cpu
(
 kaapi_address_space_id_t kasid_dest,
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_address_space_id_t kasid_src, 
 const void* src,
 const kaapi_memory_view_t* view_src
)
{
  kaapi_cuda_proc_t* cu_proc = NULL;
  int error = EINVAL;

  if ((dest ==0) || (src == 0)) return EINVAL;

  cu_proc = get_cu_context(kasid_src);
  if (cu_proc == NULL) return EINVAL;

  error = kaapi_memory_write_view
    (dest, view_dest, src, view_src, (write_func_t)memcpy_cu_dtoh, (void*)cu_proc);

  put_cu_context(cu_proc);

  return error;
}

static int kaapi_memory_write_cpu2cu
(
 kaapi_address_space_id_t kasid_dest,
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_address_space_id_t kasid_src, 
 const void* src,
 const kaapi_memory_view_t* view_src
)
{
  kaapi_cuda_proc_t* cu_proc = NULL;
  int error = EINVAL;

  if ((dest ==0) || (src == 0)) return EINVAL;

  cu_proc = get_cu_context(kasid_dest);
  if (cu_proc == NULL) return EINVAL;

  error = kaapi_memory_write_view
    (dest, view_dest, src, view_src, (write_func_t)memcpy_cu_htod, (void*)cu_proc);

  put_cu_context(cu_proc);

  return error;
}

#endif /* KAAPI_USE_CUDA */


/** Address space are 64 bits identifier decomposed in (from higher bit to lower):
    - 8 bits: type/architecture
    - 24 bits (16777216 values): for gid
    - 32 bits for user defined identifier
*/
#define KAAPI_ASID_MASK_USER  0x00000000FFFFFFFFUL
#define KAAPI_ASID_MASK_GID   0x00FFFFFF00000000UL
#define KAAPI_ASID_MASK_ARCH  0xFF00000000000000UL
kaapi_address_space_id_t kaapi_memory_address_space_create(int tid, kaapi_globalid_t _gid, int _type, size_t size )
{
  kaapi_address_space_t* retval = (kaapi_address_space_t*)malloc(sizeof(kaapi_address_space_t));
  retval->asid    = 0;
  kaapi_network_get_seginfo( retval, _gid );
  if (retval->segsize > size) retval->segsize = size;
  
#if 0
  printf("%i::[kaapi_memory_address_space_create] tid:%i, gid:%u  -> (@:%p, size:%lu)\n", 
    kaapi_network_get_current_globalid(),
    tid,
    _gid,
    (void*)retval->segaddr,
    retval->segsize
  );
#endif

  uint64_t gid;
  uint64_t type;
  /* keep 32 bits for user identifier */
  uint64_t asid = ((unsigned int)tid);

  /* extra bit used by the user: */
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_type & ~0xFF));
#else  
  if (_type & ~0xFF) return retval;
#endif
  type = (uint64_t)_type;
  
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_gid & ~0xFFFFFF));
#else  
  if (_gid & ~0xFFFFFF) return retval;
#endif
  gid = (uint64_t)_gid;

  /* store type in 8 bits */
  asid |= (type << 56) | (gid << 32);
  retval->asid = asid;
  return retval;
}


kaapi_pointer_t kaapi_memory_allocate( 
    kaapi_address_space_id_t kasid, 
    size_t size, 
    int flag 
)
{
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* compact the view to fit the new allocated bloc */
      if (flag & KAAPI_MEM_LOCAL) 
      {
        kaapi_assert_debug( kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid() );
        return (kaapi_pointer_t)malloc(size);
      }
#if defined(KAAPI_USE_NETWORK)
      if (flag & KAAPI_MEM_SHARABLE) 
      {
#if defined(KAAPI_ADDRSPACE_ISOADDRESS)
        uintptr_t segaddr = kasid->segaddr;
        /* test if enough space is possible */
        if (segaddr + size >= kasid->segsize) return 0;

        /* allocate with alignment over double boundary */
        kasid->segaddr = (kasid->segaddr + size + sizeof(double)) & ~(sizeof(double)-1);

        /* in case of local sharable memory, translate segaddr to virtual address space 
           else, the rdma instruction will translate it when communication will be posted
        */   
        if (kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid())
          ptr = (kaapi_pointer_t)kaapi_network_rdma2vas(segaddr, size);
        else 
          ptr = (kaapi_pointer_t)segaddr;

#else // KAAPI_ADDRSPACE_ISOADDRESS

        /* in case of local sharable memory do network allocation
        */   
        kaapi_assert_debug(kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid());
        ptr = (kaapi_pointer_t)kaapi_network_allocate_rdma(size);
#endif

#if 0
        printf("%i:[kaapi_memory_allocate] memory allocate @:%p, size:%lu, asid_src:", 
            kaapi_network_get_current_globalid(),
            (void*)ptr,
            size );
        kaapi_memory_address_space_fprintf( stdout, kasid );
        printf("\n");
        fflush(stdout);
#endif
        return ptr;
      }
#else // KAAPI_USE_NETWORK
      if (flag & KAAPI_MEM_SHARABLE)
        return (kaapi_pointer_t)malloc(size);
#endif
    } break;

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      /* todo: wont work on multigpu, need asid to proc */

      kaapi_cuda_proc_t* const cu_proc = get_cu_context(kasid);
      CUdeviceptr devptr;
      int error;

      if (cu_proc == NULL) return 0;
      error = allocate_cu_mem(&devptr, size);
      put_cu_context(cu_proc);

      if (error == -1) return 0;
      return (kaapi_pointer_t)devptr;

    } break;
#endif
    default:
      return 0;
  }
  return 0;
}


/** 
*/
kaapi_pointer_t kaapi_memory_allocate_view( 
  kaapi_address_space_id_t kasid, 
  kaapi_memory_view_t* view, 
  int flag 
)
{
  size_t size = kaapi_memory_view_size( view );
  
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* compact the view to fit the new allocated bloc */
      kaapi_memory_view_reallocated(view);
      return kaapi_memory_allocate( kasid, size, flag );
    } break;

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      kaapi_memory_view_reallocated(view);
      return kaapi_memory_allocate(kasid, size, flag);
    } break ;
#endif

    default:
      return 0;
  }
  return 0;
}


void kaapi_memory_global_barrier(void)
{
  kaapi_mem_barrier();
#if defined(KAAPI_USE_NETWORK)
  kaapi_network_barrier();
#endif  
}


/**
*/

static int memcpy_wrapper
(void* dummy, void* dst, const void* src, size_t size)
{
  memcpy(dst, src, size);
  return 0;
}

static int kaapi_memory_write2cpu
(
 kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
 const void* src, const kaapi_memory_view_t* view_src
)
{
  if ((dest ==0) || (src == 0)) return EINVAL;
  return kaapi_memory_write_view
    (dest, view_dest, src, view_src, memcpy_wrapper, NULL);
}


int kaapi_memory_copy( 
  kaapi_address_space_id_t kasid_dest, kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  kaapi_address_space_id_t kasid_src, const kaapi_pointer_t src, const kaapi_memory_view_t* view_src 
)
{
  int type_dest;
  int type_src;
  kaapi_globalid_t dest_gid;
  kaapi_globalid_t localgid = kaapi_network_get_current_globalid();

#if 0
  printf("[kaapi_memory_copy] copy dest@:%p, src@:%p, size:%lu\n", (void*)dest, (void*)src, kaapi_memory_view_size(view_src)); 
  fflush(stdout);
#endif

  /* cannot initiate copy if source is not local */
  if (kaapi_memory_address_space_getgid(kasid_src) != localgid)
  {
    printf("Error: not local source gid\n");
    fflush(stdout);
    return EINVAL;
  }
    
  type_dest = kaapi_memory_address_space_gettype(kasid_dest);
  type_src  = kaapi_memory_address_space_gettype(kasid_src);
  dest_gid  = kaapi_memory_address_space_getgid(kasid_dest);

  switch (type_dest) 
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      switch (type_src) 
      {
        case KAAPI_MEM_TYPE_CPU: 
          if (dest_gid == localgid)
            return kaapi_memory_write2cpu( dest, view_dest, (const void*)src, view_src ); 
          else 
#if defined(KAAPI_USE_NETWORK)
          {
#if 0
            printf("Write DMA to:%i\n", dest_gid);
            fflush(stdout);
#endif
            return kaapi_network_rdma( dest_gid, dest, view_dest, (const void*)src, view_src ); 
          }
#else
            return EINVAL;
#endif
          break;

#if defined(KAAPI_USE_CUDA)
        case KAAPI_MEM_TYPE_CUDA:
	{
	  return kaapi_memory_write_cu2cpu
	    ( kasid_dest, dest, view_dest, kasid_src, (const void*)src, view_src ); 
	} break ;
#endif /* KAAPI_USE_CUDA */

        default:
          return EINVAL;
      }
    } return 0;

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      return kaapi_memory_write_cpu2cu
	( kasid_dest, dest, view_dest, kasid_src, (const void*)src, view_src ); 
    } break ;
#endif
    
    default:
      /* bad architecture, unknown */
      KAAPI_DEBUG_INST( printf("Unknown remote address space architecture\n") );
      return EINVAL;
  }
}


/**
*/
int kaapi_memory_asyncopy( 
  kaapi_handle_t request,
  kaapi_address_space_id_t kasid_dest, void* dest, const kaapi_memory_view_t* view_dest,
  kaapi_address_space_id_t kasid_src, const void* src, const kaapi_memory_view_t* view_src 
)
{
  return EPERM;
}
