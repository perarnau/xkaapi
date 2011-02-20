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
#if defined(KAAPI_USE_NETWORK)
#include "kaapi_network.h"
#endif

/* basic function to write contiguous block of memory */
typedef int (*contiguous_write_func_t)(void*, void*, const void*, size_t);


/* Generic memcpy 
   The function takes into arguments the dest and src informations
   to made copy as well as the basic memcpy for contiguous memory region.
   Used by cuda memcpy code or 
*/
static int kaapi_memory_write_view
(
   kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
   const void* src, const kaapi_memory_view_t* view_src,
   contiguous_write_func_t write_fn, void* write_arg
)
{
  int error = EINVAL;
  
  switch (view_src->type)
  {
    case KAAPI_MEMORY_VIEW_1D:
    {
      if (view_dest->type != KAAPI_MEMORY_VIEW_1D) goto on_error;
      if (view_dest->size[0] != view_src->size[0]) goto on_error;
      error = write_fn(write_arg, kaapi_pointer2void(dest), src, view_src->size[0]*view_src->wordsize );
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
      raddr = (char*)kaapi_pointer2void(dest);
      
      if (kaapi_memory_view_iscontiguous(view_src) &&
          kaapi_memory_view_iscontiguous(view_dest))
      {
        error = write_fn(write_arg, raddr, laddr, size * view_src->wordsize);
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
          error = write_fn(write_arg, raddr, laddr, size_row);
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
#include "../cuda/kaapi_cuda_kasid.h"
#include "../machine/cuda/kaapi_cuda_error.h"

static inline kaapi_cuda_proc_t* kasid_to_cuda_proc(kaapi_address_space_id_t kasid)
{
  kaapi_cuda_proc_t* const cu_proc =
  kaapi_cuda_get_proc_by_kasid(kasid);
  kaapi_assert_debug(cu_proc);
  return cu_proc;
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
  (dest, view_dest, src, view_src, (contiguous_write_func_t)memcpy_cu_dtoh, (void*)cu_proc);
  
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
  (dest, view_dest, src, view_src, (contiguous_write_func_t)memcpy_cu_htod, (void*)cu_proc);
  
  put_cu_context(cu_proc);
  
  return error;
}


static int kaapi_memory_write_cu2cu
(
 kaapi_address_space_id_t kasid_dest,
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_address_space_id_t kasid_src, 
 const void* src,
 const kaapi_memory_view_t* view_src
 )
{
  /* use the host to make the tmp copy for now.
   assume dest_size == src_size.
   assume correctness regarding the tmp view
   todo: cuMemcpyDtoD(). warning, async function
   */
  
  int error = 0;
  
  kaapi_cuda_proc_t* cu_proc;
  
  const size_t size = kaapi_memory_view_size(view_dest);
  void* tmp_buffer = malloc(size);
  if (tmp_buffer == NULL) return EINVAL;
  
  /* cu2cpu(tmp, src) */
  cu_proc = get_cu_context(kasid_src);
  if (cu_proc == NULL) goto on_error;
  error = kaapi_memory_write_view
  ((kaapi_pointer_t)tmp_buffer, view_dest, src, view_src, (contiguous_write_func_t)memcpy_cu_dtoh, (void*)cu_proc);
  put_cu_context(cu_proc);
  if (error) goto on_error;
  
  /* cpu2cu(dst, tmp) */
  cu_proc = get_cu_context(kasid_dest);
  if (cu_proc == NULL) goto on_error;
  error = kaapi_memory_write_view
  (dest, view_dest, tmp_buffer, view_dest, (contiguous_write_func_t)memcpy_cu_htod, (void*)cu_proc);
  put_cu_context(cu_proc);
  if (error) goto on_error;
  
on_error:
  free(tmp_buffer);
  return error;
}

#endif /* KAAPI_USE_CUDA */



static int memcpy_wrapper( void* arg, void* dest, const void* src, size_t size )
{
  memcpy( dest, src, size );
  return 0;
}


/* CPU to CPU copy
*/
static int kaapi_memory_write_cpu2cpu
(
  kaapi_pointer_t dest,
  const kaapi_memory_view_t* view_dest,
  const void* src,
  const kaapi_memory_view_t* view_src
)
{
  if (kaapi_pointer_isnull(dest) || (src == 0)) return EINVAL;
  return 
    kaapi_memory_write_view
      (dest, view_dest, src, view_src, memcpy_wrapper, NULL);
}


/** Main entry point:
    - depending of source and destination, then call the appropriate function
    The choice is hard coded into table:
    type: KAAPI_MEM_TYPE_CPU   0x1
     or : KAAPI_MEM_TYPE_CUDA  0x2
    source: must be local.
    dest: local or remote. Only remote cpu is supported.
*/

typedef int (*kaapi_signature_memcpy_func_t)( 
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 const void* src,
 const kaapi_memory_view_t* view_src
);

#if !defined(KAAPI_USE_CUDA)
#define kaapi_memory_write_cpu2cu 0
#define kaapi_memory_write_cu2cpu 0
#define kaapi_memory_write_cu2cu  0
#endif
#if !defined(KAAPI_USE_NETWORK)
#define kaapi_memory_write_rdma 0
#else
static int kaapi_memory_write_rdma(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 const void* src,
 const kaapi_memory_view_t* view_src
)
{
  return kaapi_network_rdma(
    kaapi_pointer2gid(dest), 
    dest, view_dest,
    src, view_src 
  );
}
#endif

/* switcher: the_good_choice_is[local][dest_type][src_type]
   local: 0 or 1
   dest_type= 0 -> CPU, 1->GPU
   src_type= 0 -> CPU, 1->GPU
*/
static kaapi_signature_memcpy_func_t
  the_good_choice_is[2][2][2] = 
{
  /* the_good_choice_is[0]: local copy */
  {
    /* the_good_choice_is[0][0]: local copy to CPU */
    { kaapi_memory_write_cpu2cpu, kaapi_memory_write_cu2cpu },

    /* the_good_choice_is[0][1]: local copy to GPU */
    { kaapi_memory_write_cpu2cu, kaapi_memory_write_cu2cu },
  },
  
  /* the_good_choice_is[1]: remote copy */
  {
    /* the_good_choice_is[0][0]: local copy to CPU */
    { kaapi_memory_write_rdma, 0 },

    /* the_good_choice_is[0][1]: local copy to GPU */
    { 0, 0 }
  }
};


/**/
int kaapi_memory_copy( 
  kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  const kaapi_pointer_t src, const kaapi_memory_view_t* view_src 
)
{
  int type_dest;
  int type_src;
  kaapi_globalid_t dest_gid;
  kaapi_globalid_t localgid = kaapi_network_get_current_globalid();
  kaapi_signature_memcpy_func_t fnc;
  
#if 0
  printf("[kaapi_memory_copy] copy dest@:%p, src@:%p, size:%lu\n", (void*)dest, (void*)src, kaapi_memory_view_size(view_src)); 
  fflush(stdout);
#endif
  
  /* cannot initiate copy if source is not local */
  if (kaapi_pointer2gid(src) != localgid)
  {
    printf("Error: not local source gid\n");
    fflush(stdout);
    return EINVAL;
  }
  
  type_dest = kaapi_memory_address_space_gettype(kaapi_pointer2asid(dest))-1;
  type_src  = kaapi_memory_address_space_gettype(kaapi_pointer2asid(src))-1;
  dest_gid  = kaapi_pointer2gid(dest);
  
  fnc = the_good_choice_is[(dest_gid == localgid ? 0: 1)][type_dest][type_src];
  return fnc( dest, view_dest, kaapi_pointer2void(src), view_src );
}


/**
 */
int kaapi_memory_asyncopy( 
  kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  const kaapi_pointer_t src, const kaapi_memory_view_t* view_src,
  void (*callback)(void*), void* argcallback
)
{
  kaapi_memory_copy(dest, view_dest, src, view_src );
  if (callback !=0) 
    callback(argcallback);
  return 0;
}
