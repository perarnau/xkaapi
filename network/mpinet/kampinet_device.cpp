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
#include "kampinet_device.h"
#include "kampinet_channel.h"
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

namespace MPINET {


// --------------------------------------------------------------------
Device::Device( )
 : Net::Device("mpinet")
{ }


// --------------------------------------------------------------------
Device::~Device()
{
}


// --------------------------------------------------------------------
int Device::initialize()
{
  int argc  = 1;
  const char* argv[] = {"merde de mpi"};
  int err;
  int provided;
  err= MPI_Init_thread(&argc, (char***)&argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE)
  {
    printf("****Warning the MPI implementation may not work with multithreaded process\n");
  }
  
  _openchannel.clear();
  
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
  return 0;
}


// --------------------------------------------------------------------
int Device::terminate()
{
  int err;
  _openchannel.clear();
  err = MPI_Finalize();
  if (err != MPI_SUCCESS) return EINVAL;
  return 0;
}


// --------------------------------------------------------------------
int Device::abort()
{
  int err;
  err = MPI_Abort( MPI_COMM_WORLD, EINTR );
  if (err != MPI_SUCCESS) return EINVAL;
  return 0;
}


// --------------------------------------------------------------------
Net::OutChannel* Device::open_channel( const char* url )
{
  int dest;
  OutChannel* channel;
  
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
  std::map<int,OutChannel*>::iterator icurr = _openchannel.find(dest);
  if (icurr != _openchannel.end()) return icurr->second;
  channel = new MPINET::OutChannel();
  if (0 != channel->initialize()) return 0;
  _openchannel.insert( std::make_pair(dest, channel) );
  return channel;
}


// --------------------------------------------------------------------
int Device::close_channel(Net::OutChannel* ch)
{
  MPINET::OutChannel* channel = dynamic_cast<MPINET::OutChannel*> (ch);
  std::map<int,OutChannel*>::iterator icurr = _openchannel.find(channel->_dest);
  if (icurr == _openchannel.end()) return ENOENT;
  if (icurr->second != channel) return EINVAL;
  _openchannel.erase( icurr );
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
