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

/**
*/
kaapi_pointer_t kaapi_memory_allocate( 
    const kaapi_address_space_id_t kasid,
    const size_t size,
    const int flag
)
{
  kaapi_pointer_t retval = kaapi_make_pointer(kasid, 0);
  
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* compact the view to fit the new allocated bloc */
      if (flag & KAAPI_MEM_LOCAL) 
      {
        kaapi_assert_debug( kaapi_memory_address_space_getgid(kasid) == kaapi_network_get_current_globalid() );
        retval.ptr = (uintptr_t)malloc(size+15);
        retval.ptr &= ~0xF;
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
    }
      break;

#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      retval.ptr = kaapi_cuda_mem_alloc(kasid, size, (kaapi_access_mode_t)flag);
    }
      break;
#endif /* KAAPI_USE_CUDA */

    default:
    {
      retval = kaapi_make_nullpointer();
      return retval;
    }
  }
  return retval;
}


/** 
*/
kaapi_pointer_t kaapi_memory_allocate_view( 
  const kaapi_address_space_id_t kasid,
  kaapi_memory_view_t* view, 
  const int flag
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
      kaapi_cuda_mem_free(ptr);
      return 0;
    }
#endif
    
    default:
      return EINVAL;
  }
  return 0;
}

int kaapi_memory_increase_access_view(
                             const kaapi_address_space_id_t kasid,
                             kaapi_pointer_t* const ptr,
                             kaapi_memory_view_t* const view,
                             const int flag
                             )
{
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* do nothing for now */
    };
      
#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      kaapi_cuda_mem_inc_use(ptr, view, (kaapi_access_mode_t)flag);
    }
#endif
      
    default:
    {
    }
  }
  
  return 0;
}

int kaapi_memory_decrease_access_view(
                                      const kaapi_address_space_id_t kasid,
                                      kaapi_pointer_t* const ptr,
                                      kaapi_memory_view_t* const view,
                                      const int flag
                                      )
{
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* do nothing for now */
    };
      
#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      kaapi_cuda_mem_dec_use(ptr, view, (kaapi_access_mode_t)flag);
    }
#endif
      
    default:
    {
    }
  }
  
  return 0;
}
