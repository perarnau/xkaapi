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
#include "kaapi_memory.h"

#if defined(KAAPI_USE_CUDA)
#include "kaapi_cuda.h"
#endif

/** Address space are 64 bits identifier decomposed in (from higher bit to lower):
    - 8 bits: type/architecture
    - 24 bits (16777216 values): for gid
    - 32 bits for user defined identifier
*/
#define KAAPI_ASID_MASK_USER  0x00000000FFFFFFFFUL
#define KAAPI_ASID_MASK_GID   0x00FFFFFF00000000UL
#define KAAPI_ASID_MASK_ARCH  0xFF00000000000000UL
kaapi_address_space_t kaapi_memory_address_space_create(int tid, kaapi_globalid_t _gid, int _type )
{
  kaapi_address_space_t gid;
  kaapi_address_space_t type;
  /* keep 32 bits for user identifier */
  kaapi_address_space_t asid = ((unsigned int)tid);

  /* extra bit used by the user: */
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_type & ~0xFF));
#else  
  if (_type & ~0xFF) return 0;
#endif
  type = (kaapi_address_space_t)_type;
  
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(0 == (_gid & ~0xFFFFFF));
#else  
  if (_gid & ~0xFFFFFF) return 0;
#endif
  gid = (kaapi_address_space_t)_gid;

  /* store type in 8 bits */
  asid |= (type << 56) | (gid << 32);
  return asid;
}


/** 
*/
kaapi_pointer_t kaapi_memory_allocate( 
  kaapi_address_space_t kasid, 
  kaapi_memory_view_t* view, 
  int flag )
{
  size_t size = kaapi_memory_view_size( view );
  switch (kaapi_memory_address_space_gettype(kasid))
  {
    case KAAPI_MEM_TYPE_CPU:
    {
      /* compact the view to fit the new bloc */
      kaapi_memory_view_reallocated(view);
      if (flag & KAAPI_MEM_LOCAL) 
        return (kaapi_pointer_t)malloc(size);
#if defined(KAAPI_USE_NETWORK)
      if (flag & KAAPI_MEM_SHARABLE)  
        return (kaapi_pointer_t)kaapi_network_allocate_rdma(size);
#else
      if (flag & KAAPI_MEM_SHARABLE)
        return (kaapi_pointer_t)malloc(size);
#endif
    } break;
#if defined(KAAPI_USE_CUDA)
    case KAAPI_MEM_TYPE_CUDA:
    {
      /* should be able to allocate iff the current cuda context attached to a device */
      return (kaapi_pointer_t)
    }
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
static int kaapi_memory_write2cpu( 
  kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  const void* src, const kaapi_memory_view_t* view_src 
)
{
  switch (view_src->type) 
  {
    case KAAPI_MEM_VIEW_1D:
    {
      if (view_dest->type != KAAPI_MEM_VIEW_1D) return EINVAL;
      if (view_dest->size[0] != view_src->size[0]) return EINVAL;
      memcpy( (void*)dest, (const void*)src, view_src->size[0] );
      return 0;
    } break;

    case KAAPI_MEM_VIEW_2D:
    {
      const char* laddr;
      char* raddr;
      size_t size;

      if (view_dest->type != KAAPI_MEM_VIEW_2D) return EINVAL;

      if (view_dest->size[0] != view_src->size[0]) return EINVAL;
      size = view_src->size[0] * view_src->size[1];
      if (size != view_dest->size[0] * view_dest->size[1]) return EINVAL;
      
      laddr = (const char*)src;
      raddr = (char*)dest;
      
      if (kaapi_memory_view_iscontiguous(view_src) && kaapi_memory_view_iscontiguous(view_dest))
      {
          memcpy( raddr, laddr, size );
      }
      else 
      {
        kaapi_assert_debug( view_dest->size[1] == view_src->size[1] );
        size_t i;
        size_t llda;
        size_t rlda;
        llda  = view_src->lda;
        rlda  = view_dest->lda;
        for (i=0; i<view_src->size[0]; ++i, laddr += llda, raddr += rlda)
          memcpy( raddr, laddr, view_src->size[1] );
      }
      return 0;
      break;
    }
    default:
      return EINVAL;
      break;
  }
}


int kaapi_memory_copy( 
  kaapi_address_space_t kasid_dest, kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
  kaapi_address_space_t kasid_src, const kaapi_pointer_t src, const kaapi_memory_view_t* view_src 
)
{
  int type_dest;
  int type_src;
  kaapi_globalid_t dest_gid;
  kaapi_globalid_t localgid = kaapi_network_get_current_globalid();

  /* cannot initiate copy if source is not local */
  if (kaapi_memory_address_space_getgid(src) == localgid)
    return EINVAL;
    
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
            kaapi_memory_write2cpu( dest, view_dest, (const void*)src, view_src ); 
          else 
#if defined(KAAPI_USE_NETWORK)
            kaapi_network_rdma( dest_gid, dest, view_dest, (const void*)src, view_src ); 
#else
            return EINVAL;
#endif
          break;
        case KAAPI_MEM_TYPE_CUDA:
          break;
        default:
          return EINVAL;
      }
    } return 0;

    case KAAPI_MEM_TYPE_CUDA:
    {
    } return 0;
    
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
  kaapi_address_space_t kasid_dest, void* dest, const kaapi_memory_view_t* view_dest,
  kaapi_address_space_t kasid_src, const void* src, const kaapi_memory_view_t* view_src 
)
{
  return EPERM;
}
