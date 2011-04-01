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
#include "kanet_network.h"
#include "kataknet_channel.h"
#include "kataknet_device.h"
#include "kanet_types.h"
#include <iostream>

namespace TAKNET {


// --------------------------------------------------------------------
void* Device::skeleton( void* arg )
{
  Device* device = (Device*)arg;
  device->skel();
  return 0;
}


// --------------------------------------------------------------------
int Device::skel( )
{
  char buffer_incom_msg[1024];
  int err;
  timeval ts;
  ts.tv_sec  = 2;
  ts.tv_usec = 0;
  while (!_ack_term)
  {
    unsigned long taktukrank;
    size_t msg_size = 0;
    err = taktuk_wait_message( &taktukrank, &msg_size,  &ts);
    if (err == TAKTUK_ETMOUT)
    {
      if (_ack_term) return 0;
      continue;
    }
    if (err ==TAKTUK_EFCLSD) return TAKTUK_EFCLSD;
    /* assume null size == eclose ? */
    if (msg_size ==0) continue;
    TAKNET_SAFE( err );
    
    Header* header = (Header*)buffer_incom_msg;
    err = taktuk_read( buffer_incom_msg, msg_size );
    if (err ==TAKTUK_EFCLSD) return TAKTUK_EFCLSD;
    TAKNET_SAFE( err );
    std::cout << "[taknet] receive RDMA header | [AM header + AM data], size:" << msg_size << std::endl;

    switch (header->opcode)
    {
      case 'A': /* active message */
      {
        kaapi_assert( header->size < 512 );
        ka::Service_fnc service = ka::Network::resolve_service( (uint8_t)header->sid );
        std::cout << "[taknet] fnc @:" << (void*)header->ptr << ", AM data size:" << header->size << std::endl;
        (*service)(err, taktukrank, buffer_incom_msg+sizeof(Header), header->size );
      } break;
      case 'D': /* rdma request */
      {
        char* local_addr = (char*)header->ptr;
        if ( ((uintptr_t)local_addr < (uintptr_t)_segaddr) || 
             ((uintptr_t)local_addr+header->size >= (uintptr_t)_segaddr+_segsize))
        { /* outof bound of the segment: drop every thing */
          kaapi_assert(0);
        }
        else {
          /* read a second the message containing the data */
          unsigned long from;
          size_t size;
          std::cout << "[taknet] RDMA data size:" << header->size << std::endl;
          err = taktuk_recv( &from, local_addr, &size, 0 );
          TAKNET_SAFE( err );
          kaapi_assert(from == taktukrank);
          kaapi_assert(size == header->size);
        }
      } break;
      case 'T': /* abort on termination */
        _ack_term = true;
    }
  }
  return 0;
}


} // -namespace
