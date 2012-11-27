/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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

#if defined(KAAPI_USE_NETWORK)
#include "kaapi_network.h"
#endif

/* basic function to write contiguous block of memory */
typedef int (*contiguous_write_func_t)(void*, void*, const void*, size_t);

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
      
      error = write_fn(write_arg, kaapi_pointer2void(dest), kaapi_pointer2void(src), view_src->size[0]*view_src->wordsize );
      break;
    }
      
    case KAAPI_MEMORY_VIEW_2D:
    {
      const char* laddr;
      char* raddr;
      size_t size;
      
      kaapi_assert(view_dest->type == KAAPI_MEMORY_VIEW_2D);
      if (view_dest->size[0] != view_src->size[0])
      /* this case may be implemented */
        return EINVAL;
      
      size = view_src->size[0] * view_src->size[1];
      if (view_src->size[1] != view_dest->size[1])
        return EINVAL;
      
      laddr = kaapi_pointer2void(src);
      raddr = (char*)kaapi_pointer2void(dest);
      
      if (kaapi_memory_view_iscontiguous(view_src) &&
          kaapi_memory_view_iscontiguous(view_dest))
      {
        error = write_fn(write_arg, raddr, laddr, size * view_src->wordsize);
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
          if (error) break;
        }
      }
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

static inline int kaapi_memory_copy_cuda2cpu
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
 )
{
  /* this thread is the source pointer (CUDA) */
  if(kaapi_memory_map_get_current_asid() == kaapi_pointer2asid(src))
    return kaapi_cuda_mem_copy_dtoh(dest, view_dest, src, view_src);
  
  /* copy performed by a CPU thread from GPU memory */
  kaapi_processor_id_t src_kid = kaapi_memory_map_asid2kid(kaapi_pointer2asid(src));
  return kaapi_cuda_mem_copy_dtoh_from_host(dest, view_dest, src, view_src, src_kid);
}

static int kaapi_memory_copy_cpu2cuda
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
 )
{
  return kaapi_cuda_mem_copy_htod(dest, view_dest, src, view_src);
}


static int kaapi_memory_copy_cuda2cuda
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
 )
{
  fprintf(stdout, "%s:%d:%s: ERROR kid=%d src=@%p dst=@%p size=%lu: not implemented\n",
          __FILE__, __LINE__, __FUNCTION__,
          kaapi_get_self_kid(),
          kaapi_pointer2void(dest),
          kaapi_pointer2void(src),
          kaapi_memory_view_size(view_src)
          );
  fflush(stdout);
  kaapi_abort();
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
static int kaapi_memory_copy_cpu2cpu
(
 kaapi_pointer_t dest,
 const kaapi_memory_view_t* view_dest,
 kaapi_pointer_t src,
 const kaapi_memory_view_t* view_src
 )
{
  if (kaapi_pointer_isnull(dest) || kaapi_pointer_isnull(src))
    return EINVAL;
  return kaapi_memory_write_view(dest, view_dest, src, view_src, memcpy_wrapper, NULL);
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
#define kaapi_memory_write_cpu2cuda 0
#define kaapi_memory_write_cuda2cpu 0
#define kaapi_memory_write_cuda2cuda  0
#endif

#if !defined(KAAPI_USE_NETWORK)
#define kaapi_memory_write_rdma 0
#else
static inline int kaapi_memory_write_rdma(
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
    { kaapi_memory_copy_cpu2cpu, kaapi_memory_copy_cuda2cpu },
    
    /* the_good_choice_is[0][1]: local copy to GPU */
    { kaapi_memory_copy_cpu2cuda, kaapi_memory_copy_cuda2cuda },
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
  
#if 0
  //#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%d copy dest=@%p, src=@%p, size=%lu dst_gid=%d src_gid=%d (%d -> %d)\n",
          __FUNCTION__,
          kaapi_get_self_kid(),
          (void*)dest.ptr, (void*)src.ptr, kaapi_memory_view_size(view_src), kaapi_pointer2gid(dest), kaapi_pointer2gid(src),
          type_src, type_dest);
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
