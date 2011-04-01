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
#include "kataknet_device.h"
#include "kataknet_channel.h"
#include "kanet_network.h"
#include "ka_init.h"
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <stdint.h>
#include <sys/mman.h>
#include <iostream>
#include <inttypes.h>

namespace TAKNET {

// --------------------------------------------------------------------

#define KAAPI_SEGSZ_REQUEST  TAKNET_PAGESIZE
#define KAAPI_MINHEAPOFFSET 4096*128
#define TAKNET_PAGESIZE getpagesize()

// --------------------------------------------------------------------
Device::Device( )
 : ka::Device("taknet"),
   _wcom_rank(0),
   _wcom_size(0),
   _ack_term(false)
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

  _ack_term = false;
  
  err = taktuk_init_threads();
  TAKNET_SAFE( err );
  
  /* get rank, count */
  err = taktuk_get("rank", &_wcom_rank );
  TAKNET_SAFE( err );
  --_wcom_rank;
  err = taktuk_get("count", &_wcom_size );
  TAKNET_SAFE( err );
  
  /* segment size: default 4MBytes */
  uintptr_t seg_size = 4194304;
  const char* sseg_size = getenv("KAAPI_NETWORK_SEGMENT");
  if (sseg_size !=0)
  {
    seg_size = atoi(sseg_size);
  }

  /* base address: should be computed online ? */
  const char* sseg_addr = getenv("KAAPI_NETWORK_BASE_ADDR");
  void* seg_addr = (void*)0x7FAF65D49000ULL; /* arbitrary .... */
  if (sseg_addr !=0)
    seg_addr = (void*)strtoull(sseg_addr, 0, 16); /* base 16 conversion */

  /* round to multiple of TAKNET_PAGESIZE */
  seg_size = ((seg_size + TAKNET_PAGESIZE-1)/TAKNET_PAGESIZE) * TAKNET_PAGESIZE;
  seg_addr = mmap(seg_addr, seg_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, (off_t)0 );
  kaapi_assert( seg_addr != (void*)-1 );
  _segsize = seg_size;
  _segaddr = seg_addr;

  /* Get base address for all segments (taknet configure is TAKNET_ALIGN_SEGMENTS ==1) */
  _segsp   = 0;

#if defined(KAAPI_DEBUG)
  std::cout << _wcom_rank << "::[taknet] #nodes :" << _wcom_size << std::endl;
  std::cout << _wcom_rank << "::[taknet] seginfo @:" << _segaddr
            << ", size:" << _segsize
            << std::endl;
#endif
  kaapi_assert( pthread_mutex_init(&_posix_mutex, 0) == 0);
  kaapi_assert( pthread_cond_init(&_posix_cond, 0) == 0);
  
  /* register taknet specific services */
  kaapi_assert( 0 == ka::Network::register_service( service_term_id,       &service_term ) );
  kaapi_assert( 0 == ka::Network::register_service( service_barrier_id,    &service_barrier ) );
  kaapi_assert( 0 == ka::Network::register_service( service_ackbarrier_id, &service_ackbarrier ));
  _state.write(S_INIT);
  
  /* start thread to wait into incomming message */
  err = pthread_create(&_tid, 0, &Device::skeleton, this);
  kaapi_assert( err == 0);

  return 0;
}


// --------------------------------------------------------------------
int Device::commit()
{
  /* declare all nodes */
  for (unsigned long i=0; i<_wcom_size; ++i)
  {
    std::ostringstream surl;
    surl << get_name() << ":" << i;
    ka::Network::object.add_node(i, surl.str().c_str());
  }
  
  ka::Init::set_local_gid( _wcom_rank );
  
  return 0;
}


// --------------------------------------------------------------------
void Device::service_term(int err, ka::GlobalId source, void* buffer, size_t sz_buffer )
{
}


// --------------------------------------------------------------------
int Device::terminate()
{
  int err;
  
  barrier();

  if (_wcom_rank ==0)
  {
    for (unsigned long i=1; i<_wcom_size; ++i)
    {
      TAKNET::OutChannel* channel = (TAKNET::OutChannel*)ka::Network::object.get_default_local_route(i);
      Header header;
      header.opcode   = 'T';
      header.size     = 0;
      err = taktuk_send( channel->_dest+1, TAKTUK_TARGET_ANY, &header, sizeof(Header) );
    }
    _ack_term = true;
  }

#if 0 /* here cannot send a message to myself... */
  /* wakeup dispatcher thread */
  Header header;
  header.opcode   = 'T';
  header.size     = 0;
//  err = taktuk_send( _wcom_rank+1, TAKTUK_TARGET_ANY, &header, sizeof(Header) );
//  kaapi_assert_debug(err);
#endif

//  std::cout << "[taknet] Wait end of dispatcher thread" << std::endl;
  
  err = pthread_join(_tid, 0);
  kaapi_assert(err ==0);
//  std::cout << "[taknet] End of dispatcher thread ok" << std::endl;

  _state.write(S_TERM);
  
  taktuk_leave_threads();
  
  /* umap the memory */
  munmap( _segaddr, _segsize );
  
  return 0;
}


// --------------------------------------------------------------------
int Device::finalize()
{
  _state.write(S_ERROR);
#if defined(KAAPI_DEBUG)
  std::cerr << _wcom_rank << "::Abort" << std::endl;
#endif
  taktuk_leave_threads( );
  return 0;
}


// --------------------------------------------------------------------
void Device::poll() 
{
#if 0
  if (_wcom_rank !=0)
  {
    printf("%i::[taknet] poll\n", _wcom_rank); fflush(stdout);
  }
#endif
}


// --------------------------------------------------------------------
ka::OutChannel* Device::open_channel( const char* url )
{
  unsigned long dest;
  OutChannel* channel;
  
  if (_state.read() != S_INIT) return 0;
  
  /* format: taknet:rank */
  if (url ==0) return 0;
  if (strncmp(url, "taknet:", 7) != 0) return 0;
  url += 7;
  /* verify that all remainder chars are digits */
  int i = 0;
  if (url[i] ==0) return 0;
  while (url[i] !=0) if (!isdigit(url[i++])) return 0;
  dest = atoi( url );
  if ((dest <0) || (dest > _wcom_size)) return 0;

  channel = new TAKNET::OutChannel();
  if (channel->initialize(dest) !=0) return 0;
  channel->set_device( this );

  return channel;
}


// --------------------------------------------------------------------
int Device::close_channel(ka::OutChannel* ch)
{
  TAKNET::OutChannel* channel = dynamic_cast<TAKNET::OutChannel*> (ch);
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
