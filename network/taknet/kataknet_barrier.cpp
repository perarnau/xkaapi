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
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <stdint.h>
#include <sys/mman.h>
#include <iostream>
#include <inttypes.h>

namespace TAKNET {


// --------------------------------------------------------------------
size_t Device::_waiting = 0;   /* number of process that have reach the barrier */
size_t Device::_count   = 0;   /* number of process that have reach the barrier */
size_t Device::_recv    = 0;   /* number of ack from barrier */
pthread_mutex_t Device::_posix_mutex;
pthread_cond_t  Device::_posix_cond;

// --------------------------------------------------------------------
void Device::service_barrier(int err, ka::GlobalId source, void* buffer, size_t sz_buffer )
{
  pthread_mutex_lock(&_posix_mutex);
  ++_count;
  if ((_waiting !=0) && (_count % _waiting ==0))
    pthread_cond_signal(&_posix_cond);
  pthread_mutex_unlock(&_posix_mutex);
}

// --------------------------------------------------------------------
void Device::service_ackbarrier(int err, ka::GlobalId source, void* buffer, size_t sz_buffer )
{
  pthread_mutex_lock(&_posix_mutex);
  _recv = 1;
  pthread_cond_signal(&_posix_cond);
  pthread_mutex_unlock(&_posix_mutex);
}


// --------------------------------------------------------------------
void Device::barrier() 
{
  std::cout << "[taknet] begin barrier, @serv_barrier:" << (void*) &Device::service_barrier 
            << ", @serv_ackbarrier:" << (void*)&Device::service_ackbarrier
            << std::endl;
  if (_wcom_rank ==0)
  {
    pthread_mutex_lock(&_posix_mutex);
    ++_count;
    _waiting = _wcom_size;
    while (_count % _wcom_size !=0)
    {
      pthread_cond_wait(&_posix_cond, &_posix_mutex);
    }
    _waiting = 0;
    for (unsigned long i=1; i<_wcom_size; ++i)
    {
      ka::OutChannel* channel = ka::Network::object.get_default_local_route(i);
      channel->insert_am( Device::service_ackbarrier_id, 0, 0 );
      channel->sync();
    }
    pthread_mutex_unlock(&_posix_mutex);
  }
  else 
  {
    ka::OutChannel* channel = ka::Network::object.get_default_local_route(0);
    channel->insert_am( Device::service_barrier_id, 0, 0 );
    channel->sync();
    pthread_mutex_lock(&_posix_mutex);
    while (_recv ==0)
    {
      pthread_cond_wait(&_posix_cond, &_posix_mutex);
    }
    _recv = 0;
    pthread_mutex_unlock(&_posix_mutex);
  }
  std::cout << "[taknet] end barrier" << std::endl;
}


}