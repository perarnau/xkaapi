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

#if defined(KAAPI_USE_CUDA)
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
#endif

/**
*/
kaapi_pointer_t kaapi_memory_allocate( 
    kaapi_address_space_id_t kasid, 
    size_t size, 
    int flag 
)
{
  kaapi_pointer_t retval;
  retval.asid = kasid;
  retval.ptr  = 0;
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* compact the view to fit the new allocated bloc */
      if (flag & KAAPI_MEM_LOCAL) 
      {
        kaapi_assert_debug( kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid() );
        retval.ptr = (uintptr_t)malloc(size);
      }
#if defined(KAAPI_USE_NETWORK)
      if (flag & KAAPI_MEM_SHARABLE) 
      {
#if defined(KAAPI_ADDRSPACE_ISOADDRESS)
#else // KAAPI_ADDRSPACE_ISOADDRESS

        /* in case of local sharable memory do network allocation
        */   
        kaapi_assert_debug(kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid());
        retval.ptr = (uintptr_t)kaapi_network_allocate_rdma(size);
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
        return retval;
      }
#else // KAAPI_USE_NETWORK
      if (flag & KAAPI_MEM_SHARABLE)
        return kaapi_make_pointer(kasid, malloc(size) );
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
      retval.ptr = (uintptr_t)devptr;
      return retval;

    } break;
#endif
    default:
    {
      retval.asid = 0;
      retval.ptr  = 0;
      return retval;
    }
  }
  return retval;
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
    };

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      kaapi_memory_view_reallocated(view);
      return kaapi_memory_allocate(kasid, size, flag);
    }
#endif

    default:
    {
      kaapi_pointer_t retval = {0,0};
      return retval;
    }
  }
}


/**
*/
int kaapi_memory_deallocate( 
  kaapi_pointer_t ptr 
)
{
  if (kaapi_pointer2gid(ptr) != kaapi_network_get_current_globalid())
  {
    return EINVAL;
  }
  switch( kaapi_memory_address_space_gettype(kaapi_pointer2asid(ptr)))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      free( (void*)ptr.ptr );
      return 0;
    };

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      free_cu_mem( (CUdeviceptr) ptr.ptr );
      return 0;
    }
#endif
    
    default:
      return EINVAL;
  }
  return 0;
}

