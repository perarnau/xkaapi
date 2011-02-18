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

#if defined(KAAPI_USE_NETWORK)
#include "kanet_network.h"
#endif

extern "C" { /* minimal C interface to network */

#if defined(KAAPI_USE_NETWORK)

// --------------------------------------------------------------------
int kaapi_network_init(int* argc, char*** argv)
{
  /*
  */
  ka::Network::object.initialize(argc, argv);
  ka::Network::object.commit();
    
  return 0;
}


// --------------------------------------------------------------------
int kaapi_network_finalize()
{
  /*
  */
  ka::Network::object.terminate();
  return 0;
}


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
  uint32_t nodecount= (uint32_t)ka::Network::object.size();
  if (nodecount == 0) return 1;
  return nodecount;
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


/**
*/
void kaapi_network_barrier(void)
{
  ka::Network::object.barrier();
}


/** 
*/
void* kaapi_network_allocate_rdma(size_t size)
{
#if 0
  printf("%i:[kaapi_memory_allocate] IN memory allocate\n", ka::Network::object.local_gid());
  fflush(stdout);
#endif
  void* ptr = ka::Network::object.allocate(size);
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
    case KAAPI_MEMORY_VIEW_1D:
    {
      if (view_dest->type != KAAPI_MEMORY_VIEW_1D) return EINVAL;
      if (view_dest->size[0] != view_src->size[0]) return EINVAL;

      ka::OutChannel* channel = ka::Network::object.get_default_local_route( gid_dest );
      if (channel == 0) return EINVAL;

      channel->insert_rwdma( kaapi_pointer2uintptr(dest), src, view_src->size[0]*view_src->wordsize );
      channel->sync();

      return 0;
    } break;

    case KAAPI_MEMORY_VIEW_2D:
    {
      uintptr_t laddr; /* local address */
      kaapi_pointer_t raddr; /* remote addr */
      size_t size;

      if (view_dest->type != KAAPI_MEMORY_VIEW_2D) return EINVAL;

      if (view_dest->size[0] != view_src->size[0]) return EINVAL;
      size = view_src->size[0] * view_src->size[1];
      if (size != view_dest->size[0] * view_dest->size[1]) return EINVAL;
      
      ka::OutChannel* channel = ka::Network::object.get_default_local_route( gid_dest );
      if (channel == 0) return EINVAL;

      laddr = (uintptr_t)src;
      raddr = (kaapi_pointer_t)dest;
      
      if (kaapi_memory_view_iscontiguous(view_src) && kaapi_memory_view_iscontiguous(view_dest))
      {
        channel->insert_rwdma( kaapi_pointer2uintptr(raddr), (const void*)laddr, size*view_src->wordsize );
      }
      else 
      {
        kaapi_assert_debug( view_dest->size[1] == view_src->size[1] );
        size_t i;
        size_t size_row = view_src->size[1]*view_src->wordsize;
        size_t llda     = view_src->lda*view_src->wordsize;
        size_t rlda     = view_dest->lda*view_src->wordsize;
        for (i=0; i<view_src->size[0]; ++i, laddr += llda, raddr.ptr += rlda)
          channel->insert_rwdma( kaapi_pointer2uintptr(raddr), (const void*)laddr, size_row );
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


#else // if defined(KAAPI_USE_NETWORK)

/* This is dummy function to avoid dependencies with C interface to network
*/
/**
*/
int kaapi_network_init(int* argc, char*** argv)
{  
  return 0;
}

/**
*/
int kaapi_network_finalize()
{  
  return 0;
}


/**
*/
kaapi_globalid_t kaapi_network_get_current_globalid(void)
{
  return 0;
}

/**
*/
uint32_t kaapi_network_get_count(void)
{
  return 1;
}


void kaapi_network_poll()
{
}

void kaapi_network_barrier(void)
{
}

int kaapi_network_get_seginfo( kaapi_address_space_t* retval, kaapi_globalid_t gid )
{
  retval->segaddr = 0;
  retval->segsize = (size_t)-1;
  return 0;
}


#endif // KAAPI_USE_NETWORK

} // end extern C
