/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "kanet_network.h"

extern "C" { /* minimal C interface to network */

#if defined(KAAPI_USE_NETWORK)

/**
*/
kaapi_globalid_t kaapi_network_get_current_globalid(void)
{
  return ka::Network::object.local_gid();
}

/**
*/
uint32_t kaapi_network_get_count(void)
{
  return (uint32_t)ka::Network::object.size();
}

/**
*/
int kaapi_network_get_seginfo( kaapi_address_space_t* retval, kaapi_globalid_t gid )
{
  ka::SegmentInfo seginfo = ka::Network::object.get_seginfo( gid );
  retval->segaddr = seginfo.segaddr;
  retval->segsize = seginfo.segsize;
  return 0;
}

/**
*/
void kaapi_network_poll()
{
  ka::Network::object.poll();
}


/** Return a pointer in a memory region which is rdmable
*/
kaapi_pointer_t kaapi_network_rdma2vas(kaapi_pointer_t addr, size_t size)
{
  return (kaapi_pointer_t)ka::Network::object.bind(addr, size);
}


/**
*/
void kaapi_network_barrier(void)
{
  ka::Network::object.barrier();
}


/** 
*/
kaapi_pointer_t kaapi_network_allocate_rdma(size_t size)
{
#if 0
  printf("%i:[kaapi_memory_allocate] IN memory allocate\n", ka::Network::object.local_gid());
  fflush(stdout);
#endif
  kaapi_pointer_t ptr = (kaapi_pointer_t)ka::Network::object.allocate(size);
#if 0
  printf("%i:[kaapi_memory_allocate] OUT memory allocate @:%p, size:%lu\n", 
      ka::Network::object.local_gid(),
      (void*)ptr,
      size 
  );
#endif
  return ptr;
}


/**
*/
int kaapi_network_rdma(
  kaapi_globalid_t gid_dest, 
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

      ka::OutChannel* channel = ka::Network::object.get_default_local_route( gid_dest );
      if (channel == 0) return EINVAL;

      channel->insert_rwdma( dest, src, view_src->size[0] );
      channel->sync();

      return 0;
    } break;

    case KAAPI_MEM_VIEW_2D:
    {
      kaapi_pointer_t laddr; /* local address */
      kaapi_pointer_t raddr; /* remote addr */
      size_t size;

      if (view_dest->type != KAAPI_MEM_VIEW_2D) return EINVAL;

      if (view_dest->size[0] != view_src->size[0]) return EINVAL;
      size = view_src->size[0] * view_src->size[1];
      if (size != view_dest->size[0] * view_dest->size[1]) return EINVAL;
      
      ka::OutChannel* channel = ka::Network::object.get_default_local_route( gid_dest );
      if (channel == 0) return EINVAL;

      laddr = (kaapi_pointer_t)src;
      raddr = (kaapi_pointer_t)dest;
      
      if (kaapi_memory_view_iscontiguous(view_src) && kaapi_memory_view_iscontiguous(view_dest))
      {
        channel->insert_rwdma( raddr, (const void*)laddr, size );
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
          channel->insert_rwdma( raddr, (const void*)laddr, view_src->size[1] );
      }
      channel->sync();
      return 0;
      break;
    }
    default:
      return EINVAL;
      break;
  }
}


/**
*/
int kaapi_network_am(
  kaapi_globalid_t gid_dest, 
  kaapi_service_t service, const void* data, size_t size 
)
{
  ka::OutChannel* channel = ka::Network::object.get_default_local_route( gid_dest );
  if (channel == 0)
    return EINVAL;

  channel->insert_am( service, data, size );
  channel->sync();
  return 0;
}

#endif // KAAPI_USE_NETWORK

}
