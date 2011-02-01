/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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

/**
*/
typedef struct kaapi_exchange_addr_t {
  kaapi_comtag_t            tag;           /* tag */
  int                       tid;           /* thread identifier in the group of the receiver */
  kaapi_pointer_t           rsignal;       /* remote kaapi_comrecv_t data structure */
  kaapi_pointer_t           raddr;         /* remote data address */
  kaapi_memory_view_t       rview;         /* remote data view */
} kaapi_exchange_addr_t;


/**
*/
typedef struct kaapi_exchange_list_addr_t {
  kaapi_exchange_addr_t*     list;
  size_t                     pos;
  size_t                     size;
  struct kaapi_exchange_list_addr_t* next;
} kaapi_exchange_list_addr_t;


/** The first kaapi_exchange_addr_t is a dummy kaapi_exchange_addr_t to pass 
    the threadgroup identification.
*/
static void kaapi_threadgroup_raddrservice(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
  size_t count_addr;
  kaapi_exchange_addr_t* list = (kaapi_exchange_addr_t*)buffer;
  uint32_t grpid = (uint32_t)list->tag;
  sz_buffer -= sizeof(kaapi_exchange_addr_t);
  ++list;
  count_addr = sz_buffer / sizeof(kaapi_exchange_addr_t);

  /* resolve the thread group identifier */
  kaapi_threadgroup_t thgrp = kaapi_all_threadgroups[ grpid ];
  
#if 0
  printf("%i::[kaapi_threadgroup_raddrservice] from:%i, size: %lu, sizeof:%lu, count:%lu\n", 
      static_thgrp->localgid, 
      source, 
      sz_buffer,
      sizeof(kaapi_exchange_addr_t),
      count_addr
  );
  fflush(stdout);
#endif
  
  for (size_t i=0; i<count_addr; ++i)
  {
#if 0
    printf("%i::[kaapi_threadgroup_raddrservice] from:%i, recv tag:%i, tid:%i, rsignal:%p, raddr: %p\n",
      static_thgrp->localgid,
      source,
      (int)list[i].tag,
      (int)list[i].tid,
      (void*)list[i].rsignal,
      (void*)list[i].raddr
    );
    fflush(stdout);
#endif

    kaapi_comsend_raddr_t* com = 
        kaapi_threadgroup_findsend_tagtid( thgrp->all_sendaddr, list[i].tag, list[i].tid );

#if 0
    printf("%i::[kaapi_threadgroup_raddrservice] from:%i, find tag:%i, tid:%i, found ? %s\n",
      static_thgrp->localgid,
      source,
      (int)list[i].tag,
      (int)list[i].tid,
      (com == 0 ? "no" : "yes")
    );
    fflush(stdout);
#endif

    if (com !=0)
    {
      com->rsignal = list[i].rsignal;
      com->raddr   = list[i].raddr;
      com->rview   = list[i].rview;
    }
  }
}

/**
*/
static int kaapi_threadgroup_resolved_for( kaapi_threadgroup_t thgrp, int tid, kaapi_comlink_t* cl )
{
  kaapi_comsend_t* com;
  kaapi_comsend_raddr_t* lraddr;
  while (cl != 0)
  {
    com = cl->u.send;
    lraddr = &com->front;
    while (lraddr !=0)
    {
      if (lraddr->rsignal ==0)
      {
        int rtid = kaapi_threadgroup_asid2tid( thgrp, lraddr->asid );
        kaapi_comrecv_t* recv = kaapi_recvcomlist_find_tag( thgrp->lists_recv[rtid], com->vertag );
        if (recv != 0)
        {
          lraddr->rsignal = (kaapi_pointer_t)recv;
          lraddr->raddr   = (kaapi_pointer_t)recv->data;
          lraddr->rview   = recv->view;
        }
        else {
          printf("Tag resolution failed for asid:%llu\n", com->vertag );
#if defined(KAAPI_ADDRSPACE_ISOADDRESS)
          exit(1);
#else
#endif
        }
      }
      lraddr = lraddr->next;
    }
    cl = cl->next;
  }
  return 0;
}

#if defined(KAAPI_USE_NETWORK)
/**
*/
static int kaapi_threadgroup_update_recv( 
    kaapi_threadgroup_t thgrp, 
    kaapi_exchange_list_addr_t* array2gid,
    int tid, kaapi_comlink_t* cl 
)
{
  while (cl !=0)
  {
    kaapi_comrecv_t* curr = cl->u.recv;
    kaapi_globalid_t from = curr->from;
    if (from != thgrp->localgid)
    {
      size_t pos = array2gid[from].pos;
      array2gid[from].list[pos].tag     = curr->tag;
      array2gid[from].list[pos].tid     = curr->tid;
      array2gid[from].list[pos].rsignal = (kaapi_pointer_t)curr;
      array2gid[from].list[pos].raddr   = (kaapi_pointer_t)curr->data;
      array2gid[from].list[pos].rview   = curr->view;

#if 0
      printf("%i::[kaapi_threadgroup_update_recv] send to from:%i, that I need tag:%i, tid:%i, rsignal:%p, raddr: %p\n",
        thgrp->localgid,
        from,
        (int)array2gid[from].list[pos].tag,
        (int)array2gid[from].list[pos].tid,
        (void*)array2gid[from].list[pos].rsignal,
        (void*)array2gid[from].list[pos].raddr
      );
      fflush(stdout);
#endif
      ++pos;
      if (pos >= array2gid[from].size)
      {
        kaapi_assert_debug( pos == array2gid[from].size );
        /* double size of the array */
        size_t newsize = 2* array2gid[from].size;
        kaapi_exchange_addr_t* newarray = 
          (kaapi_exchange_addr_t*)malloc( newsize * sizeof(kaapi_exchange_addr_t) );
        kaapi_assert_debug( newarray != 0 );
        
        memcpy( newarray, array2gid[from].list, array2gid[from].size*sizeof(kaapi_exchange_addr_t) );

        free(array2gid[from].list);

        array2gid[from].list = newarray;
        array2gid[from].size = newsize;
      }

      kaapi_assert( ++pos < array2gid[from].size );

      array2gid[from].pos = pos;
    }

    cl = cl->next;
  }
  return 0;
}
#endif

int kaapi_threadgroup_barrier_partition( kaapi_threadgroup_t thgrp )
{
#if 0
  printf("%i::[kaapi_threadgroup_barrier_partition] begin\n", thgrp->localgid);
#endif
#if defined(KAAPI_USE_NETWORK)
  kaapi_memory_global_barrier();
  
  kaapi_exchange_list_addr_t* array2gid 
      = (kaapi_exchange_list_addr_t*)malloc( kaapi_network_get_count()* sizeof(kaapi_exchange_list_addr_t) );
  for (size_t i=0; i<kaapi_network_get_count(); ++i)
  {
    array2gid[i].list = malloc( 4096 );
    array2gid[i].pos  = 1; /* 0 is reserved for threadgroup identification */
    array2gid[i].size = 4096/sizeof(kaapi_exchange_addr_t);    
  }
  
  /* build the list of remote address to exchange to each node */
  for (int tid=-1; tid < thgrp->group_size; ++tid)
  {
    if (kaapi_threadgroup_tid2gid( thgrp, tid ) == thgrp->localgid)
    {
      kaapi_threadgroup_update_recv( thgrp, array2gid, tid, thgrp->lists_recv[tid] );
    }
  }


  /* exchange remote addr between every participating nodes */

  size_t localgid = thgrp->localgid;
  for (size_t gid=0; gid<kaapi_network_get_count(); ++gid)
  {
    if (gid == localgid) continue;

    if (array2gid[gid].pos >1) /* ok need to send data to other sender */
    {
      /* fill pos=0 with threadgroup identification */
      array2gid[gid].list[0].tag     = thgrp->grpid;
#if defined(KAAPI_DEBUG)        
      array2gid[gid].list[0].tid     = -10; /* marker */
      array2gid[gid].list[0].rsignal = 0;   /* marker */
      array2gid[gid].list[0].raddr   = 0;   /* marker */
      kaapi_memory_view_clear(&array2gid[gid].list[0].rview);
#endif // if defined(KAAPI_DEBUG)        
#if 0
      printf("%lu::[send raddr] to:%lu, size: %lu\n", localgid, gid, array2gid[gid].pos );
      for (size_t k=1; k<array2gid[gid].pos; ++k)
      {
        printf("\ttag:%i, tid:%i, rsignal: %p, raddr: %p, rsize: %lu\n",
           (int)array2gid[gid].list[k].tag,
           (int)array2gid[gid].list[k].tid,
           (void*)array2gid[gid].list[k].rsignal,
           (void*)array2gid[gid].list[k].raddr,
           kaapi_memory_view_size(&array2gid[gid].list[k].rview)
        );
      }
#endif

      /* send active message to gid */
/* WARNING HACK */
{      
      size_t sz_msg = array2gid[gid].pos*sizeof( kaapi_exchange_addr_t);
      kaapi_exchange_addr_t* curr = array2gid[gid].list;
      while (sz_msg>0)
      {
        /* pack packed of size 512Bytes (limit due to gasnet) */
        size_t sz_bloc = (sz_msg >= 512 ? 512 / sizeof( kaapi_exchange_addr_t) : sz_msg/sizeof( kaapi_exchange_addr_t) );
        kaapi_network_am( 
            (int)gid, 
            kaapi_threadgroup_raddrservice, 
            curr,
            sz_bloc*sizeof( kaapi_exchange_addr_t) 
        );
        if (sz_bloc == 1) 
          break;
        sz_msg -= (sz_bloc-1)*sizeof( kaapi_exchange_addr_t);
        curr += sz_bloc-1;
        /* first bloc is thread group ident */
        curr->tag     = thgrp->grpid;
#if defined(KAAPI_DEBUG)        
        curr->tid     = -10; /* marker */
        curr->rsignal = 0;   /* marker */
        curr->raddr   = 0;   /* marker */
        kaapi_memory_view_clear(&curr->rview);
#endif // if defined(KAAPI_DEBUG)        
      }
} /* END WARNING HACK */
      
      free(array2gid[gid].list);
    }
  }
  free(array2gid);
  
  /* all messages received after that ? */
  kaapi_memory_global_barrier();
#endif

  /* report remote address to send list */
  for (int tid=-1; tid<thgrp->group_size; ++tid)
  {
    if (kaapi_threadgroup_tid2gid( thgrp, tid ) == thgrp->localgid)
    {
      kaapi_threadgroup_resolved_for( thgrp, tid, thgrp->lists_send[tid] );
    }
  }


  /* this version is only for multicore machine */
  if ((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0)
    kaapi_threadgroup_save(thgrp);

  /* barrier between every nodes:
     - after the barrier any nodes may write the correct location using remote dma
  */
  kaapi_memory_global_barrier();
#if 0
  printf("%i::[kaapi_threadgroup_barrier_partition] end\n", thgrp->localgid);
#endif
  return 0;
}