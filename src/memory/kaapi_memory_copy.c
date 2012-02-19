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

#if defined(KAAPI_USE_CUDA)

#include "../cuda/kaapi_cuda.h"
#include "../machine/cuda/kaapi_cuda_mem.h"
#include "../machine/cuda/kaapi_cuda_ctx.h"
#include "../machine/cuda/kaapi_cuda_dev.h"
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
   kaapi_pointer_t src, const kaapi_memory_view_t* view_src,
   contiguous_write_func_t write_fn, void* write_arg
)
{
  int error = 0;
  
  switch (view_src->type)
  {
    case KAAPI_MEMORY_VIEW_1D:
    {
      kaapi_assert(view_dest->type == KAAPI_MEMORY_VIEW_1D);
      if (view_dest->size[0] != view_src->size[0]) 
        return EINVAL;

      error = kaapi_cuda_mem_1dcopy_dtoh( dest, view_dest, src, view_src );
      break;
    } 
      
    case KAAPI_MEMORY_VIEW_2D:
    {
      error = kaapi_cuda_mem_2dcopy_dtoh( dest, view_dest, src, view_src );

#if KAAPI_VERBOSE
  kaapi_processor_t* const proc = kaapi_get_current_processor();
  printf("[%s]: [%u:%u] (%lx:%lx -> %lx:%lx) src lda=%lu %lux%lu dst lda=%lu %lux%lu\n",
	 __FUNCTION__,
	 proc->kid, proc->proc_type,
	 src.asid, (uintptr_t)src.ptr,
	 dest.asid, (uintptr_t)dest.ptr,
	view_src->lda,
	view_src->size[0],
	view_src->size[1],
	view_dest->lda,
	view_dest->size[0],
	view_dest->size[1]
	 );
#endif 
	      break;
    }
      
    default: 
    {
      error = EINVAL;
    } break;
  }  
  return error;
}



#if defined(KAAPI_USE_CUDA)

#if 0
static inline unsigned int is_self_cu_proc
(kaapi_cuda_proc_t* cu_proc)
{
  if (cu_proc == &kaapi_get_current_processor()->cuda_proc)
    return 1;
  return 0;
}
#endif

/*
 * TODO: better way to perform GPU -> CPU copies but from the CPU 
 * assuming that we have more than 1 GPU and only the pointer, not the device
 * So, the function has to: 1) search device 2) push ctx
 */
kaapi_processor_t*
get_cu_context( void )
{
  kaapi_processor_t* cu_proc;
  if( kaapi_get_current_processor()->proc_type != KAAPI_PROC_TYPE_CUDA )
	  cu_proc= kaapi_cuda_mem_get_proc();
  else
  	 cu_proc = kaapi_get_current_processor();
  
  
  return cu_proc;
}

static kaapi_processor_t*
xxx_get_cu_context( kaapi_address_space_id_t asid )
{
	kaapi_processor_t* proc = kaapi_cuda_get_proc_by_asid( asid );
	return proc;
}

static void xxx_put_cu_context( kaapi_processor_t* proc )
{
}

void put_cu_context( kaapi_processor_t* cu_proc)
{
}

static inline int kaapi_memory_write_cu2cpu
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
)
{
  kaapi_processor_t* cu_proc = NULL;
  int error = EINVAL;

  if (kaapi_pointer_isnull(dest)) return EINVAL;
  else if (kaapi_pointer_isnull(src)) return EINVAL;
  
  //cu_proc = get_cu_context();
  cu_proc = xxx_get_cu_context( kaapi_pointer2asid(src) );
  if (cu_proc == NULL) return EINVAL;
  
  error = kaapi_memory_write_view
  (dest, view_dest, src, view_src, NULL, (void*)cu_proc);
  
  xxx_put_cu_context( cu_proc );
  
  return error;
}

static int kaapi_memory_write_cpu2cu
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
)
{
  kaapi_processor_t* cu_proc = NULL;
  int error = EINVAL;

  if (kaapi_pointer_isnull(dest)) return EINVAL;
  else if (kaapi_pointer_isnull(src)) return EINVAL;
  
  cu_proc = get_cu_context();
  if (cu_proc == NULL) return EINVAL;
  
  error = kaapi_memory_write_view
  (dest, view_dest, src, view_src, NULL, (void*)cu_proc);
  
  put_cu_context(cu_proc);
  
  return error;
}


static int kaapi_memory_write_cu2cu
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
)
{
  /* use the host to make the tmp copy for now.
   assume dest_size == src_size.
   assume correctness regarding the tmp view
   todo: cuMemcpyDtoD(). warning, async function
   */
#if 0
  int error = 0;
  kaapi_pointer_t tmp_pointer;
  
  kaapi_cuda_proc_t* cu_proc;
  
  const size_t size = kaapi_memory_view_size(view_dest);
  void* tmp_buffer = malloc(size);
  if (tmp_buffer == NULL) return EINVAL;

  tmp_pointer = kaapi_make_pointer(0, tmp_buffer);
  
  /* cu2cpu(tmp, src) */
  cu_proc = get_cu_context();
  if (cu_proc == NULL) goto on_error;

  error = kaapi_memory_write_view
    (tmp_pointer, view_dest, src, view_src,
     (contiguous_write_func_t)memcpy_cu_dtoh, (void*)cu_proc);

  put_cu_context(cu_proc);

  if (error) goto on_error;
  
  /* cpu2cu(dst, tmp) */
  cu_proc = get_cu_context();
  if (cu_proc == NULL) goto on_error;

  error = kaapi_memory_write_view
    (dest, view_dest, tmp_buffer, view_dest,
     (contiguous_write_func_t)memcpy_cu_htod, (void*)cu_proc);

  put_cu_context(cu_proc);

  if (error) goto on_error;
  
on_error:
  free(tmp_buffer);
  return error;
#endif
  return -1;
}

#endif /* KAAPI_USE_CUDA */



static int memcpy_wrapper( void* arg, void* dest, const void* src, size_t size )
{
  memcpy( dest, src, size );
  return 0;
}


/* CPU to CPU copy
*/
int kaapi_memory_write_cpu2cpu
(
  kaapi_pointer_t dest,
  const kaapi_memory_view_t* view_dest,
  kaapi_pointer_t src,
  const kaapi_memory_view_t* view_src
)
{
  if (kaapi_pointer_isnull(dest) || (kaapi_pointer_isnull(src) == 0)) return EINVAL;
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
 kaapi_pointer_t src,
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
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
)
{
  return kaapi_network_rdma(
    kaapi_pointer2gid(dest), 
    dest, view_dest,
    kaapi_pointer2void(src), view_src 
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
  

#if !defined(KAAPI_USE_CUDA)  
  /* cannot initiate copy if source is not local */
  if (kaapi_pointer2gid(src) != localgid)
  {
    printf("Error: not local source gid\n");
    fflush(stdout);
    return EINVAL;
  }
#endif
  
  type_dest = kaapi_memory_address_space_gettype(kaapi_pointer2asid(dest))-1;
  type_src  = kaapi_memory_address_space_gettype(kaapi_pointer2asid(src))-1;
  dest_gid  = kaapi_pointer2gid(dest);

#if 1 /* to_fix: we enter here with kasid == 0... */
  if (type_dest == -1) ++type_dest;
  if (type_src == -1) ++type_src;
#endif

#if 1
//#if KAAPI_VERBOSE
  fprintf(stdout, "[kaapi_memory_copy] copy dest@:%p, src@:%p, size:%lu dst_gid=%d src_gid=%d\n",
	 (void*)dest.ptr, (void*)src.ptr, kaapi_memory_view_size(view_src), kaapi_pointer2gid(dest), kaapi_pointer2gid(src) ); 
  fflush(stdout);
#endif

  fnc = the_good_choice_is[(dest_gid == localgid ? 0: 1)][type_dest][type_src];
  return fnc( dest, view_dest, src, view_src );
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
