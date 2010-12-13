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
#include "kaapi++"
#include "kampinet_device.h"
#include "kampinet_channel.h"
#include "kanet_network.h"
#include "ka_init.h"
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

namespace MPINET {


// --------------------------------------------------------------------
Device::Device( )
 : ka::Device("mpinet"),
   _comm(MPI_COMM_WORLD),
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
int Device::initialize()
{
  int err;
#if 0 
  err = MPI_Init(&ka::System::saved_argc, (char***)&ka::System::saved_argv);
#else
  int provided;
  err= MPI_Init_thread(&ka::System::saved_argc, (char***)&ka::System::saved_argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE)
  {
    std::cerr << "****Warning the MPI implementation may not work with multithreaded process" << std::endl;
  }
#endif
  kaapi_assert( err == MPI_SUCCESS);
  _comm = MPI_COMM_WORLD;    
  _state.write(S_INIT);

  err = pthread_create(&_tid, 0, &Device::skeleton, this);
  kaapi_assert(err ==0);

  return 0;
}


// --------------------------------------------------------------------
int Device::commit()
{
  int err;
  err = MPI_Comm_size( MPI_COMM_WORLD, &_wcom_size );
  if (err != MPI_SUCCESS) return EINVAL;
  err = MPI_Comm_rank( MPI_COMM_WORLD, &_wcom_rank );
  if (err != MPI_SUCCESS) return EINVAL;
  
  /* declare all nodes */
  for (int i=0; i<_wcom_size; ++i)
  {
    std::ostringstream surl;
    surl << get_name() << ":" << i;
    ka::Network::object.add_node(i, surl.str().c_str());
  }
  
  ka::Init::set_local_gid( _wcom_rank );
  return 0;
}


// --------------------------------------------------------------------
int Device::terminate()
{
  int err;
  printf("%i::Device should stop\n", ka::System::local_gid);
  fflush(stdout);
  
  printf("%i::all devices have reach the barrier\n", ka::System::local_gid);
  fflush(stdout);
  
  if (ka::System::local_gid ==0)
  {
    for (int i=1; i<_wcom_size; ++i)
    {
      ka::OutChannel* channel = ka::Network::object.get_default_local_route(i);
      kaapi_assert( channel != 0 );
      channel->insert_am( &service_term, 0, 0);
      channel->sync();
      printf("%i::Send term message to:%s\n", ka::System::local_gid,  channel->get_peer_url());
      fflush(stdout);
    }
  }
  printf("%i::all devices have reach the barrier\n", ka::System::local_gid);
  fflush(stdout);
  
  err = pthread_join(_tid, 0);
  kaapi_assert(err ==0);
  _state.write(S_TERM);
  
  err = MPI_Finalize();
  if (err != MPI_SUCCESS) return EINVAL;
  return 0;
}


// --------------------------------------------------------------------
int Device::abort()
{
  int err;
  _state.write(S_ERROR);
  err = MPI_Abort( MPI_COMM_WORLD, EINTR );
  if (err != MPI_SUCCESS) return EINVAL;
  return 0;
}


// --------------------------------------------------------------------
ka::OutChannel* Device::open_channel( const char* url )
{
  int dest;
  OutChannel* channel;
  
  if (_state.read() != S_INIT) return 0;
  
  /* format: mpinet:rank */
  if (url ==0) return 0;
  if (strncmp(url, "mpinet:", 7) != 0) return 0;
  url += 7;
  /* verify that all remainder chars are digits */
  int i = 0;
  if (url[i] ==0) return 0;
  while (url[i] !=0) if (!isdigit(url[i++])) return 0;
  dest = atoi( url );
  if ((dest <0) || (dest > _wcom_size)) return 0;

  channel = new MPINET::OutChannel();
  if (channel->initialize(_comm, dest) !=0) return 0;
  channel->set_device( this );

  return channel;
}


// --------------------------------------------------------------------
int Device::close_channel(ka::OutChannel* ch)
{
  MPINET::OutChannel* channel = dynamic_cast<MPINET::OutChannel*> (ch);
  channel->terminate();
  delete channel;
  return 0;
}


// --------------------------------------------------------------------
const char* Device::get_urlconnect( ) const
{
  /* cannot connect MPI process */
  return 0;
}

} // - namespace Net...
