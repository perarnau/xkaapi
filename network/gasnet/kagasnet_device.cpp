/*
** xkaapi
** 
** Copyright 2010 INRIA.
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
#include "kagasnet_device.h"
#include "kagasnet_channel.h"
#include "kanet_network.h"
#include "ka_init.h"
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <stdint.h>

#include <iostream>

namespace GASNET {

// --------------------------------------------------------------------

#define KAAPI_SEGSZ_REQUEST  GASNET_PAGESIZE
#define KAAPI_MINHEAPOFFSET 4096*128

// --------------------------------------------------------------------
Device::Device( )
 : ka::Device("gasnet"),
   _wcom_rank(0),
   _wcom_size(0),
   _ack_term(0)
{ 
  _state.write(S_CREATE);
}


// --------------------------------------------------------------------
Device::~Device()
{
}

// --------------------------------------------------------------------
int Device::initialize(int* argc, char*** argv)
{
  int err;

  _ack_term = 0;
  
  gasnet_handlerentry_t htable[] = { 
    { kaapi_gasnet_service_call_id,     (void (*)())kaapi_gasnet_service_call  }
  };
  
  err = gasnet_init(argc, argv);
  GASNET_SAFE( err );
  
  /* segment size: default 4MBytes */
  uintptr_t seg_size = 4194304;
  const char* sseg_size = gasnet_getenv("KAAPI_NETWORK_SEGMENT");
  if (sseg_size !=0)
  {
    seg_size = atoi(sseg_size);
    
  }

  /* multiple of GASNET_PAGESIZE */
  seg_size = ((seg_size + GASNET_PAGESIZE-1)/GASNET_PAGESIZE) * GASNET_PAGESIZE;

  GASNET_SAFE(gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t),
                            seg_size, KAAPI_MINHEAPOFFSET));

  /* Get base address for all segments (gasnet configure is GASNET_ALIGN_SEGMENTS ==1) */
  _seginfo = new gasnet_seginfo_t[gasnet_nodes()];
  gasnet_getSegmentInfo( _seginfo, gasnet_nodes());
  _segaddr = _seginfo[gasnet_mynode()].addr;
  _segsize = _seginfo[gasnet_mynode()].size;
  _segsp   = 0;

#if defined(KAAPI_DEBUG)
  std::cout << gasnet_mynode() << "::[gasnet] #nodes :" << gasnet_nodes() << std::endl;
  std::cout << gasnet_mynode() << "::[gasnet] seginfo @:" << _segaddr
            << ", size:" << _segsize
            << std::endl;
#endif
  
  _state.write(S_INIT);

//  err = pthread_create(&_tid, 0, &Device::skeleton, this);
//  kaapi_assert(err ==0);

  return 0;
}


// --------------------------------------------------------------------
int Device::commit()
{
  _wcom_rank = gasnet_mynode();
  _wcom_size = gasnet_nodes();
  
  /* declare all nodes */
  for (int i=0; i<_wcom_size; ++i)
  {
    std::ostringstream surl;
    surl << get_name() << ":" << i;
    ka::Network::object.add_node(i, surl.str().c_str());
  }
  
  ka::Init::set_local_gid( _wcom_rank );
  
  barrier();

  return 0;
}


// --------------------------------------------------------------------
int Device::terminate()
{
#if 0
  printf("%i::[gasnet] begin terminate\n", _wcom_rank); fflush(stdout);
#endif
  
  barrier();

  if (_wcom_rank ==0)
  {
    for (int i=1; i<_wcom_size; ++i)
    {
      ka::OutChannel* channel = ka::Network::object.get_default_local_route(i);
      kaapi_assert( channel != 0 );
      channel->sync();
    }
  }

  barrier();
#if 0
  printf("%i::[gasnet] end terminate\n", _wcom_rank); fflush(stdout);
#endif

//  err = pthread_join(_tid, 0);
//  kaapi_assert(err ==0);
  _state.write(S_TERM);
  
  gasnet_exit(0);
  return 0;
}


// --------------------------------------------------------------------
int Device::finalize()
{
  _state.write(S_ERROR);
  gasnet_exit( EINTR );
  return 0;
}


// --------------------------------------------------------------------
void Device::poll() 
{
  int err;
#if 0
  if (_wcom_rank !=0)
  {
    printf("%i::[gasnet] poll\n", _wcom_rank); fflush(stdout);
  }
#endif
  err = gasnet_AMPoll();    
  kaapi_assert( (err == GASNET_OK) );
}


// --------------------------------------------------------------------
void Device::barrier() 
{
  gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);  
  GASNET_SAFE(gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS));
}


// --------------------------------------------------------------------
ka::OutChannel* Device::open_channel( const char* url )
{
  int dest;
  OutChannel* channel;
  
  if (_state.read() != S_INIT) return 0;
  
  /* format: gasnet:rank */
  if (url ==0) return 0;
  if (strncmp(url, "gasnet:", 7) != 0) return 0;
  url += 7;
  /* verify that all remainder chars are digits */
  int i = 0;
  if (url[i] ==0) return 0;
  while (url[i] !=0) if (!isdigit(url[i++])) return 0;
  dest = atoi( url );
  if ((dest <0) || (dest > _wcom_size)) return 0;

  channel = new GASNET::OutChannel();
  if (channel->initialize(dest) !=0) return 0;
  channel->set_device( this );

  return channel;
}


// --------------------------------------------------------------------
int Device::close_channel(ka::OutChannel* ch)
{
  GASNET::OutChannel* channel = dynamic_cast<GASNET::OutChannel*> (ch);
  channel->terminate();
  delete channel;
  return 0;
}


// --------------------------------------------------------------------
const char* Device::get_urlconnect( ) const
{
  return 0;
}

} // - namespace Net...
